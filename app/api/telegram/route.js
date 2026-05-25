// ============================================================================
// app/api/telegram/route.js
// ----------------------------------------------------------------------------
// KudiGuide AI - Telegram Webhook Handler (Next.js 14 App Router, JavaScript)
//
// End-to-end pipeline:
//   1. Telegram POSTs an update to this endpoint.
//   2. We pull either the text message OR the voice file off the update.
//   3. For voice notes, we resolve the file path via Telegram's Bot API,
//      download the binary stream, wrap it as a virtual File, and ship it
//      to Groq's Whisper-large-v3 model for transcription.
//   4. The resulting transcript is fed into Groq Llama-3-70B with strict
//      JSON-mode output, using a specialized accounting system prompt that
//      extracts a structured ledger entry.
//   5. The parsed entry (plus chatId, raw transcript, timestamp) is written
//      to the root `transactions` collection in Firestore.
//   6. A friendly markdown receipt is composed and POSTed back to the user
//      via Telegram's `sendMessage` endpoint.
//
// Resilience notes:
//   - We ALWAYS return HTTP 200 to Telegram, even on internal errors. If we
//     returned a non-2xx, Telegram would relentlessly retry the same update
//     (potentially creating duplicate ledger entries when our bug is fixed).
//   - Each external call (Telegram, Groq, Firestore) is wrapped in its own
//     try/catch so a single failure mode degrades gracefully rather than
//     bringing down the whole pipeline.
// ============================================================================

import { NextResponse } from "next/server";
import Groq from "groq-sdk";
import { File } from "buffer";
import { admin, adminDb } from "../../../lib/firebaseAdmin";

// ----------------------------------------------------------------------------
// Runtime configuration
// ----------------------------------------------------------------------------
// We need the Node.js runtime (not Edge) for two reasons:
//   1. `firebase-admin` relies on Node-only APIs (fs, net, etc.).
//   2. Buffering/streaming audio uploads to Groq is far simpler under Node.
export const runtime = "nodejs";

// Avoid Next.js trying to cache POST responses — every webhook delivery is
// fresh and must execute the full handler.
export const dynamic = "force-dynamic";

// ----------------------------------------------------------------------------
// Groq client (singleton across hot reloads in dev)
// ----------------------------------------------------------------------------
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------
const TELEGRAM_API_BASE = "https://api.telegram.org";
const WHISPER_MODEL = "whisper-large-v3";
const LLAMA_MODEL = "llama-3.3-70b-versatile";

// The accounting system prompt is the heart of our extraction quality.
// It locks the model into a single role and a strict, well-defined schema.
const BOOKKEEPING_SYSTEM_PROMPT = `You are KudiGuide, a precise bookkeeping engine for small business owners in Nigeria and across Africa. Your job is to read a user message and determine their intent: either logging a business transaction OR requesting a financial report.

INTENT 1: LOGGING ("logging")
- User describes a business event (sale, expense, debt).
- Normalise currency (e.g. 5k -> 5000).
- "income": money in, "expense": money out.
- "debtor": anyone who owes money.

INTENT 2: REPORTING ("reporting")
- User asks for profit, loss, summary, or details for a timeframe.
- Timeframes: "today", "yesterday", "this week", "this month", "all time".
- Default to "today" if unspecified.

Output EXACTLY this JSON shape:
{
  "intent": "logging" | "reporting",
  "transaction_detected": true | false,
  "type": "income" | "expense" | "none",
  "amount": number | null,
  "description": "string",
  "has_debtor": true | false,
  "debtor_details": {
    "name": string | null,
    "amount_owed": number | null,
    "due_date": string | null
  },
  "report_params": {
    "timeframe": "today" | "yesterday" | "week" | "month" | "all",
    "specific_date": string | null
  }
}`;

// ============================================================================
// MAIN HANDLER
// ============================================================================
export async function POST(request) {
  try {
    // ---- 1. Parse the incoming Telegram update payload ---------------------
    // We defensively try/catch the JSON parse — a malformed body should not
    // crash the route, just return 200 so Telegram stops retrying.
    let body;
    try {
      body = await request.json();
    } catch (parseErr) {
      console.error("[telegram] Failed to parse webhook JSON:", parseErr);
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    // Telegram updates can contain many top-level fields (edited_message,
    // callback_query, channel_post, etc.). We only care about plain `message`
    // events. Anything else: ack and bail.
    const message = body?.message;
    if (!message) {
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    const chatId = message?.chat?.id;
    if (!chatId) {
      // Without a chat id we can't reply, so just acknowledge and exit.
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ---- 2. Resolve the user's textual input -------------------------------
    // Either the user typed a message (`message.text`) or sent a voice note
    // (`message.voice`). For voice notes we run the full Whisper pipeline.
    let transcript = null;
    let inputSource = "unknown";

    if (typeof message.text === "string" && message.text.trim().length > 0) {
      transcript = message.text.trim();
      inputSource = "text";

      // ---- 2a. Handle specific commands like /start -------------------------
      if (transcript.toLowerCase().startsWith("/start")) {
        await safeSendTelegramMessage(
          chatId,
          "👋 *Welcome to KudiGuide AI!* 📚\n\nI'm your voice-first financial ledger. Just send me a *voice note* or a *text* describing a business transaction, and I'll log it for you.\n\n*Examples:*\n• \"Sold 3 bags for 5000\"\n• \"Bought fuel 2500\"\n• \"Tunde took 2 crates on credit, will pay Friday\"\n\nHow can I help you today?"
        );
        return NextResponse.json({ ok: true }, { status: 200 });
      }
    } else if (message.voice && message.voice.file_id) {
      inputSource = "voice";
      try {
        transcript = await transcribeTelegramVoice(message.voice.file_id);
      } catch (voiceErr) {
        console.error("[telegram] Voice transcription failed:", voiceErr);
        await safeSendTelegramMessage(
          chatId,
          "⚠️ I couldn't hear that voice note clearly. Please try again or send a text message."
        );
        return NextResponse.json({ ok: true }, { status: 200 });
      }
    } else {
      // Unsupported message type (sticker, photo without caption, etc.).
      await safeSendTelegramMessage(
        chatId,
        "👋 Hi! Send me a *text* or a *voice note* describing a sale, expense, or credit, and I'll log it for you."
      );
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    // If after all that we still have no transcript, gracefully exit.
    if (!transcript || transcript.trim().length === 0) {
      await safeSendTelegramMessage(
        chatId,
        "🤔 I didn't catch any words in that message. Mind trying again?"
      );
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ---- 3. Run the bookkeeping extraction via Groq Llama-3 ----------------
    let parsedData;
    try {
      if (!process.env.GROQ_API_KEY) {
        throw new Error("GROQ_API_KEY is missing in environment variables.");
      }
      parsedData = await extractLedgerEntry(transcript);
    } catch (llmErr) {
      console.error("[telegram] LLM extraction failed:", llmErr.message || llmErr);
      if (llmErr.stack) console.error(llmErr.stack);
      
      await safeSendTelegramMessage(
        chatId,
        "⚠️ I had trouble understanding that. Could you rephrase it (e.g. *Sold 3 bags for 5000*)?"
      );
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ---- 4. Dispatch based on Intent (Logging vs Reporting) ----------------
    if (parsedData.intent === "reporting") {
      try {
        const report = await generateFinancialReport(chatId, parsedData.report_params);
        await safeSendTelegramMessage(chatId, report);
      } catch (reportErr) {
        console.error("[telegram] Report generation failed:", reportErr);
        await safeSendTelegramMessage(
          chatId,
          "⚠️ I couldn't pull your report right now. Please try again in a moment."
        );
      }
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ---- 5. Persist the entry to Firestore (Logging Flow) -----------------
    try {
      await adminDb.collection("transactions").add({
        chatId,
        rawTranscript: transcript,
        inputSource,
        parsedLedgerData: parsedData,
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
      });
    } catch (dbErr) {
      console.error("[telegram] Firestore write failed:", dbErr);
      await safeSendTelegramMessage(
        chatId,
        "⚠️ I understood you, but couldn't save the entry just now. Please try again in a moment."
      );
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ---- 6. Compose and send the human-readable receipt --------------------
    const receipt = formatReceipt(parsedData, transcript);
    await safeSendTelegramMessage(chatId, receipt);

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (fatalErr) {
    // Catch-all so we NEVER bubble a 500 back to Telegram (which would trigger
    // an aggressive retry storm).
    console.error("[telegram] Fatal handler error:", fatalErr);
    return NextResponse.json({ ok: true }, { status: 200 });
  }
}

// ============================================================================
// HELPERS
// ============================================================================

/**
 * Downloads a Telegram voice note and transcribes it via Groq Whisper.
 *
 * Telegram's file API is a two-step dance:
 *   1) GET  /bot<token>/getFile?file_id=... -> returns metadata containing
 *      a relative `file_path` like "voice/file_42.oga".
 *   2) GET  /file/bot<token>/<file_path>    -> serves the raw binary stream.
 *
 * We then wrap the binary as a virtual File and stream it to Groq.
 *
 * @param {string} fileId - Telegram's opaque file_id for the voice note.
 * @returns {Promise<string>} the transcribed text.
 */
async function transcribeTelegramVoice(fileId) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  if (!token) {
    throw new Error("Missing TELEGRAM_BOT_TOKEN env var.");
  }

  try {
    // Step 1: resolve file metadata -> file_path.
    const metaUrl = `${TELEGRAM_API_BASE}/bot${token}/getFile?file_id=${encodeURIComponent(
      fileId
    )}`;
    const metaResp = await fetch(metaUrl);
    if (!metaResp.ok) {
      throw new Error(`getFile failed: ${metaResp.status}`);
    }
    const metaJson = await metaResp.json();
    const filePath = metaJson?.result?.file_path;
    if (!filePath) {
      throw new Error("Missing file_path in Telegram response.");
    }

    // Step 2: download the binary audio stream.
    const fileUrl = `${TELEGRAM_API_BASE}/file/bot${token}/${filePath}`;
    const fileResp = await fetch(fileUrl);
    if (!fileResp.ok) {
      throw new Error(`File download failed: ${fileResp.status}`);
    }
    const arrayBuffer = await fileResp.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Step 3: ship it to Whisper via Groq SDK.
    // We use a explicit File object from 'buffer' to ensure compatibility
    // in the Node.js runtime. This provides the 'name' and 'type' Groq needs.
    const transcription = await groq.audio.transcriptions.create({
      file: new File([buffer], "voice.ogg", { type: "audio/ogg" }),
      model: WHISPER_MODEL,
      response_format: "json",
    });

    return (transcription?.text ?? "").trim();
  } catch (err) {
    console.error("[transcribeTelegramVoice] Error:", err.message || err);
    throw err;
  }
}

/**
 * Sends the transcript to Llama-3-70B with strict JSON-mode output and
 * returns the parsed ledger object. Includes a defensive fallback shape
 * if parsing somehow fails.
 *
 * @param {string} transcript - the user's text (typed or transcribed).
 * @returns {Promise<object>} the structured ledger entry.
 */
async function extractLedgerEntry(transcript) {
  const completion = await groq.chat.completions.create({
    model: LLAMA_MODEL,
    response_format: { type: "json_object" }, // strict JSON mode
    temperature: 0.1, // deterministic-ish for accounting
    messages: [
      { role: "system", content: BOOKKEEPING_SYSTEM_PROMPT },
      { role: "user", content: transcript },
    ],
  });

  const rawContent = completion?.choices?.[0]?.message?.content ?? "{}";

  let parsed;
  try {
    parsed = JSON.parse(rawContent);
  } catch (jsonErr) {
    console.error(
      "[telegram] LLM returned non-JSON despite json_object mode:",
      rawContent
    );
    // Fall back to a safe "nothing detected" shape rather than crashing.
    parsed = { intent: "logging", transaction_detected: false, type: "none" };
  }

  // Normalize the shape
  return {
    intent: parsed.intent === "reporting" ? "reporting" : "logging",
    transaction_detected: Boolean(parsed.transaction_detected),
    type: ["income", "expense", "none"].includes(parsed.type)
      ? parsed.type
      : "none",
    amount:
      typeof parsed.amount === "number" && Number.isFinite(parsed.amount)
        ? parsed.amount
        : null,
    description:
      typeof parsed.description === "string" && parsed.description.length > 0
        ? parsed.description
        : transcript,
    has_debtor: Boolean(parsed.has_debtor),
    debtor_details: {
      name: parsed?.debtor_details?.name ?? null,
      amount_owed:
        typeof parsed?.debtor_details?.amount_owed === "number"
          ? parsed.debtor_details.amount_owed
          : null,
      due_date: parsed?.debtor_details?.due_date ?? null,
    },
    report_params: {
      timeframe: ["today", "yesterday", "week", "month", "all"].includes(
        parsed?.report_params?.timeframe
      )
        ? parsed.report_params.timeframe
        : "today",
      specific_date: parsed?.report_params?.specific_date ?? null,
    },
  };
}

/**
 * Queries Firestore and generates a summary report for the user.
 */
async function generateFinancialReport(chatId, params) {
  const { timeframe } = params;
  const start = new Date();
  start.setHours(0, 0, 0, 0);

  if (timeframe === "yesterday") {
    start.setDate(start.getDate() - 1);
    const end = new Date(start);
    end.setHours(23, 59, 59, 999);
  } else if (timeframe === "week") {
    start.setDate(start.getDate() - 7);
  } else if (timeframe === "month") {
    start.setMonth(start.getMonth() - 1);
  } else if (timeframe === "all") {
    start.setFullYear(2000); // effectively all time
  }

  const snapshot = await adminDb
    .collection("transactions")
    .where("chatId", "==", chatId)
    .where("createdAt", ">=", admin.firestore.Timestamp.fromDate(start))
    .orderBy("createdAt", "desc")
    .get();

  if (snapshot.empty) {
    return `📝 *Report for ${capitalize(timeframe)}:*\n\nNo transactions found in this period.`;
  }

  let totalIncome = 0;
  let totalExpense = 0;
  let totalDebt = 0;
  const transactions = [];

  snapshot.forEach((doc) => {
    const data = doc.data();
    const ledger = data.parsedLedgerData;
    if (ledger.type === "income") totalIncome += ledger.amount || 0;
    if (ledger.type === "expense") totalExpense += ledger.amount || 0;
    if (ledger.has_debtor) totalDebt += ledger.debtor_details?.amount_owed || 0;
    
    transactions.push(ledger);
  });

  const profit = totalIncome - totalExpense;

  const lines = [
    `📊 *Financial Report: ${capitalize(timeframe)}*`,
    "",
    `🟢 *Total Income:* ₦${formatNumber(totalIncome)}`,
    `🔴 *Total Expense:* ₦${formatNumber(totalExpense)}`,
    "---",
    `💰 *${profit >= 0 ? "Net Profit" : "Net Loss"}:* ₦${formatNumber(Math.abs(profit))}`,
    "",
    `📒 *Outstanding Credit:* ₦${formatNumber(totalDebt)}`,
    "",
    `_Generated based on ${snapshot.size} records._`
  ];

  return lines.join("\n");
}

/**
 * Builds the markdown receipt that gets sent back to the user.
 * Uses Telegram's "Markdown" parse mode (note: NOT MarkdownV2), so the
 * supported syntax is *bold*, _italic_, `code`, and [links](url).
 *
 * @param {object} entry - the structured ledger entry.
 * @param {string} transcript - the original transcript, used as a header.
 */
function formatReceipt(entry, transcript) {
  // No financial event detected -> friendly acknowledgement.
  if (!entry.transaction_detected || entry.type === "none") {
    return [
      "👋 *Got it!*",
      "",
      `I heard: _"${escapeMarkdown(transcript)}"_`,
      "",
      "I didn't spot a clear sale, expense, or credit in that one. Try something like *Sold 4 crates for 8000* or *Bought fuel 2500*.",
    ].join("\n");
  }

  const isIncome = entry.type === "income";
  const headerEmoji = isIncome ? "🟢" : "🔴";
  const headerLabel = isIncome ? "Income Logged" : "Expense Logged";
  const amountLine =
    entry.amount !== null
      ? `*Amount:* ₦${formatNumber(entry.amount)}`
      : `*Amount:* _not specified_`;

  const lines = [
    `${headerEmoji} *${headerLabel}* ✅`,
    "",
    amountLine,
    `*Type:* ${capitalize(entry.type)}`,
    `*Details:* ${escapeMarkdown(entry.description)}`,
  ];

  // Outstanding debt block — only included when meaningful.
  if (entry.has_debtor) {
    const d = entry.debtor_details || {};
    lines.push("");
    lines.push("📒 *Debtor / Outstanding Credit*");
    lines.push(`• *Name:* ${escapeMarkdown(d.name || "Unknown")}`);
    if (d.amount_owed !== null && d.amount_owed !== undefined) {
      lines.push(`• *Amount owed:* ₦${formatNumber(d.amount_owed)}`);
    }
    if (d.due_date) {
      lines.push(`• *Due:* ${escapeMarkdown(String(d.due_date))}`);
    }
  }

  lines.push("");
  lines.push("_Saved to your KudiGuide ledger._ 📚");

  return lines.join("\n");
}

/**
 * Posts a message back to the user via Telegram's sendMessage endpoint.
 * Wrapped to swallow errors so a failed reply never bubbles back to the
 * webhook caller.
 */
async function safeSendTelegramMessage(chatId, text) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  if (!token) {
    console.error("[telegram] Cannot send message: TELEGRAM_BOT_TOKEN missing.");
    return;
  }

  try {
    const resp = await fetch(`${TELEGRAM_API_BASE}/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        text,
        parse_mode: "Markdown",
        disable_web_page_preview: true,
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "<no body>");
      console.error(
        `[telegram] sendMessage failed (${resp.status}): ${errText}`
      );
    }
  } catch (sendErr) {
    console.error("[telegram] sendMessage threw:", sendErr);
  }
}

// ----------------------------------------------------------------------------
// Small formatting utilities
// ----------------------------------------------------------------------------

function capitalize(s) {
  if (!s || typeof s !== "string") return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatNumber(n) {
  // Pretty-print with thousands separators, e.g. 25000 -> "25,000".
  try {
    return Number(n).toLocaleString("en-NG");
  } catch {
    return String(n);
  }
}

/**
 * Escapes the characters that have special meaning in Telegram's "Markdown"
 * (v1) parse mode so user-supplied text doesn't accidentally break formatting.
 * Telegram v1 markdown is forgiving — we only need to escape the active set.
 */
function escapeMarkdown(text) {
  if (typeof text !== "string") return "";
  return text.replace(/([_*`\[])/g, "\\$1");
}

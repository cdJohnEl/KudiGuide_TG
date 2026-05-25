// ============================================================================
// lib/firebaseAdmin.js
// ----------------------------------------------------------------------------
// Secure singleton initializer for the Firebase Admin SDK.
//
// Why a singleton?
//   Next.js (especially in dev mode with hot-reload) can re-evaluate modules
//   many times during a single process lifetime. Calling `admin.initializeApp`
//   more than once on the same default app throws:
//     "The default Firebase app already exists."
//   We guard with `admin.apps.length` so we only initialize once per process.
//
// Why parse the private key?
//   When you store the GCP service-account private key in an env var (e.g. in
//   Vercel, Render, .env.local), the literal newline characters typically get
//   serialized as the two-character escape sequence "\n". The Admin SDK needs
//   real newlines to parse the PEM block, so we transform them back here.
// ============================================================================

import admin from "firebase-admin";

/**
 * Lazily initialize the Firebase Admin app exactly once per Node.js process.
 * Subsequent imports of this module will reuse the already-initialized app.
 */
function getFirebaseAdminApp() {
  // If an app instance already exists in this process, reuse it.
  if (admin.apps.length > 0) {
    return admin.app();
  }

  // Pull the three required credentials from server-side environment vars.
  // These MUST be set in `.env.local` (or your hosting provider's env config).
  const projectId = process.env.FIREBASE_PROJECT_ID;
  const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
  const rawPrivateKey = process.env.FIREBASE_PRIVATE_KEY;

  // Fail loudly during boot if any required credential is missing. This is
  // much easier to debug than a silent Firestore write failure later.
  if (!projectId || !clientEmail || !rawPrivateKey) {
    throw new Error(
      "[firebaseAdmin] Missing required env vars: FIREBASE_PROJECT_ID, " +
        "FIREBASE_CLIENT_EMAIL, and/or FIREBASE_PRIVATE_KEY."
    );
  }

  // Convert escaped "\n" sequences into actual newline characters so the PEM
  // block parses correctly.
  const privateKey = rawPrivateKey.replace(/\\n/g, "\n");

  // Initialize the default app with a service-account credential.
  return admin.initializeApp({
    credential: admin.credential.cert({
      projectId,
      clientEmail,
      privateKey,
    }),
  });
}

// Initialize (or retrieve) the singleton app instance at module load time.
const firebaseApp = getFirebaseAdminApp();

// Export a ready-to-use Firestore instance plus the admin namespace itself
// (the latter is needed for things like `admin.firestore.FieldValue.serverTimestamp()`).
const adminDb = firebaseApp.firestore();

export { admin, adminDb };
export default firebaseApp;

/* =============================================================
   FaceID Portal — app.js
   Handles: page routing, model status, registration, face
   capture, login, and live face verification.
   API base: http://localhost:8000
   ============================================================= */

const API = 'http://localhost:8000';

// ── Page token storage ─────────────────────────────────────────
let sessionToken = null;   // set after successful login

// ── Active camera stream reference ────────────────────────────
let currentStream = null;

/* ─────────────────────────────────────────────────────────────
   PAGE ROUTING
   ───────────────────────────────────────────────────────────── */
function showPage(name) {
  // Stop any running camera first
  stopCameraStream();

  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

  const page = document.getElementById(`page-${name}`);
  if (!page) return;
  page.classList.add('active');

  // Page-specific init
  if (name === 'landing')    initLanding();
  if (name === 'register')   initRegister();
  if (name === 'recognize')  initRecognize();
}

function stopCameraStream() {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  // Reset capture state
  captureInterval = null;
  capturedFrames = [];
}


/* ─────────────────────────────────────────────────────────────
   LANDING
   ───────────────────────────────────────────────────────────── */
async function initLanding() {
  const dot   = document.querySelector('.status-dot');
  const label = document.getElementById('status-label');

  dot.className   = 'status-dot';  // reset
  label.textContent = 'Checking model…';

  try {
    const res  = await fetch(`${API}/api/model-status`);
    const data = await res.json();
    if (data.status === 'ready') {
      dot.classList.add('ready');
      const n = data.classes ? data.classes.length : 0;
      label.textContent = `Model ready — ${n} enrolled user${n !== 1 ? 's' : ''}`;
    } else if (data.status === 'training') {
      label.textContent = 'Model training in background…';
    } else if (data.status === 'not_trained' || data.status === 'error') {
      dot.classList.add('error');
      label.textContent = 'Model needs ≥ 2 enrolled users to train';
    } else {
      label.textContent = `Status: ${data.status}`;
    }
  } catch {
    dot.classList.add('error');
    label.textContent = 'Cannot reach API — is the server running?';
  }
}


/* ─────────────────────────────────────────────────────────────
   REGISTER — Step 1: Account Details
   ───────────────────────────────────────────────────────────── */
let registeredEmail = '';

function initRegister() {
  // Reset to Step 1
  document.getElementById('reg-step1').classList.remove('hidden');
  document.getElementById('reg-step2').classList.add('hidden');
  document.getElementById('reg-step-indicator').textContent = 'Step 1 of 2';
  document.getElementById('reg-error').classList.add('hidden');
  document.getElementById('form-register').reset();
  registeredEmail = '';
}

async function handleRegister(event) {
  event.preventDefault();

  const nameVal = document.getElementById('reg-name').value.trim();
  const email   = document.getElementById('reg-email').value.trim();
  const pass    = document.getElementById('reg-password').value;
  const errEl   = document.getElementById('reg-error');
  const btn     = document.getElementById('btn-register-submit');

  errEl.classList.add('hidden');
  setButtonLoading(btn, true);

  try {
    const res  = await fetch(`${API}/api/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_name: nameVal, email, password: pass })
    });
    const data = await res.json();

    if (!res.ok) {
      showError(errEl, data.detail || 'Registration failed.');
      return;
    }

    // Success → go to Step 2
    registeredEmail = email;
    document.getElementById('reg-step1').classList.add('hidden');
    document.getElementById('reg-step2').classList.remove('hidden');
    document.getElementById('reg-step-indicator').textContent = 'Step 2 of 2';

    // Start webcam
    await startRegisterCamera();

  } catch (e) {
    showError(errEl, 'Network error — is the API server running?');
  } finally {
    setButtonLoading(btn, false);
  }
}


/* ─────────────────────────────────────────────────────────────
   REGISTER — Step 2: Face Capture
   ───────────────────────────────────────────────────────────── */
const TOTAL_FRAMES = 30;
const FRAME_INTERVAL_MS = 500;  // capture every 500 ms

let capturedFrames = [];
let captureInterval = null;

async function startRegisterCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
    currentStream = stream;
    const video = document.getElementById('reg-video');
    video.srcObject = stream;
  } catch (e) {
    showError(document.getElementById('capture-error'), 'Camera access denied. Please allow camera permission.');
  }
}

function startCapture() {
  if (!currentStream) return;

  capturedFrames = [];
  document.getElementById('capture-error').classList.add('hidden');
  document.getElementById('capture-success').classList.add('hidden');
  document.getElementById('btn-start-capture').classList.add('hidden');
  document.getElementById('btn-stop-capture').classList.remove('hidden');

  // Activate scan line
  document.getElementById('scan-line').classList.add('active');

  updateCaptureProgress(0);

  captureInterval = setInterval(captureFrame, FRAME_INTERVAL_MS);
}

function captureFrame() {
  const video  = document.getElementById('reg-video');
  const canvas = document.getElementById('reg-canvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  const b64 = canvas.toDataURL('image/jpeg', 0.85);
  capturedFrames.push(b64);

  updateCaptureProgress(capturedFrames.length);

  if (capturedFrames.length >= TOTAL_FRAMES) {
    stopCapture();
    submitFaces();
  }
}

function stopCapture() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
  document.getElementById('scan-line').classList.remove('active');
  document.getElementById('btn-stop-capture').classList.add('hidden');
  document.getElementById('btn-start-capture').classList.remove('hidden');
}

function updateCaptureProgress(count) {
  const pct = Math.min((count / TOTAL_FRAMES) * 100, 100);
  document.getElementById('capture-progress-bar').style.width = `${pct}%`;
  document.getElementById('capture-count').textContent = `${count} / ${TOTAL_FRAMES} frames`;
}

async function submitFaces() {
  const errEl = document.getElementById('capture-error');
  const sucEl = document.getElementById('capture-success');
  errEl.classList.add('hidden');
  sucEl.classList.add('hidden');

  const btn = document.getElementById('btn-start-capture');
  btn.disabled = true;
  btn.textContent = 'Uploading…';

  try {
    const res  = await fetch(`${API}/api/capture-faces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: registeredEmail, frames: capturedFrames })
    });
    const data = await res.json();

    if (!res.ok) {
      showError(errEl, data.detail || 'Face capture failed.');
      btn.disabled = false;
      btn.textContent = 'Retry Capture';
      return;
    }

    sucEl.textContent = `✓ ${data.message}`;
    sucEl.classList.remove('hidden');

    // Stop camera, show "Go to Login" button
    stopCameraStream();
    btn.textContent = '✓ Done — Go to Sign In';
    btn.disabled = false;
    btn.onclick = () => showPage('recognize');

  } catch (e) {
    showError(errEl, 'Upload failed. Check your connection.');
    btn.disabled = false;
    btn.textContent = 'Retry Capture';
  }
}


/* ─────────────────────────────────────────────────────────────
   RECOGNIZE — Login
   ───────────────────────────────────────────────────────────── */
function initRecognize() {
  document.getElementById('login-section').classList.remove('hidden');
  document.getElementById('verify-section').classList.add('hidden');
  document.getElementById('login-error').classList.add('hidden');
  document.getElementById('form-login').reset();
  document.getElementById('verify-result').classList.add('hidden');
  document.getElementById('verify-result').innerHTML = '';
  sessionToken = null;
}

async function handleLogin(event) {
  event.preventDefault();

  const email   = document.getElementById('login-email').value.trim();
  const pass    = document.getElementById('login-password').value;
  const errEl   = document.getElementById('login-error');
  const btn     = document.getElementById('btn-login-submit');

  errEl.classList.add('hidden');
  setButtonLoading(btn, true);

  try {
    const res  = await fetch(`${API}/api/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password: pass })
    });
    const data = await res.json();

    if (!res.ok) {
      showError(errEl, data.detail || 'Login failed.');
      return;
    }

    sessionToken = data.token;
    document.getElementById('verify-greeting').textContent =
      `Hi ${data.full_name} — look straight at the camera`;

    // Move to face verify
    document.getElementById('login-section').classList.add('hidden');
    document.getElementById('verify-section').classList.remove('hidden');

    // Start webcam
    await startVerifyCamera();

  } catch (e) {
    showError(errEl, 'Network error — is the API server running?');
  } finally {
    setButtonLoading(btn, false);
  }
}


/* ─────────────────────────────────────────────────────────────
   RECOGNIZE — Face Verify
   ───────────────────────────────────────────────────────────── */
async function startVerifyCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
    currentStream = stream;
    const video = document.getElementById('verify-video');
    video.srcObject = stream;
  } catch (e) {
    document.getElementById('verify-status-msg').textContent = '⚠ Camera access denied.';
  }
}

async function verifyFace() {
  if (!currentStream || !sessionToken) return;

  const btn     = document.getElementById('btn-verify');
  const statusEl = document.getElementById('verify-status-msg');
  const resultEl = document.getElementById('verify-result');
  const scanLine = document.getElementById('verify-scan-line');

  resultEl.classList.add('hidden');
  resultEl.innerHTML = '';
  btn.disabled = true;
  statusEl.textContent = '🔍 Scanning your face…';
  scanLine.classList.add('active');

  // Capture one frame
  const video  = document.getElementById('verify-video');
  const canvas = document.getElementById('verify-canvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const frame = canvas.toDataURL('image/jpeg', 0.9);

  // Small delay so user sees the scan animation
  await delay(900);
  scanLine.classList.remove('active');

  try {
    const res  = await fetch(`${API}/api/verify-face`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${sessionToken}`
      },
      body: JSON.stringify({ frame })
    });
    const data = await res.json();

    if (!res.ok) {
      const detail = data.detail || 'Verification error — please try again.';
      renderResult(resultEl, 'warning', '⚠', detail, null, false);
      statusEl.textContent = 'Verification failed — see message below.';
      btn.disabled = false;
      return;
    }

    // data: { name, confidence, granted, match, logged_in_as }
    if (data.granted && data.match) {
      renderResult(resultEl, 'granted',
        '✅',
        `Welcome, ${data.name}!`,
        `Confidence: <strong>${data.confidence}%</strong> — Identity verified`,
        true
      );
      statusEl.innerHTML = '<span style="color:var(--green)">● Access granted</span>';
      // Session is consumed — stop camera
      stopCameraStream();
    } else if (data.granted && !data.match) {
      renderResult(resultEl, 'denied',
        '❌',
        `Face Mismatch`,
        `Detected: <strong>${data.name}</strong> (${data.confidence}%) — Does not match login credentials`,
        true
      );
      statusEl.innerHTML = '<span style="color:var(--red)">● Identity mismatch</span>';
      stopCameraStream();
    } else {
      // not granted
      renderResult(resultEl, 'denied',
        '🚫',
        'Unrecognized Face',
        `Confidence too low (${data.confidence}%) — try again in better lighting`,
        false
      );
      statusEl.textContent = 'Could not verify — retry';
      btn.disabled = false;
    }

  } catch (e) {
    renderResult(resultEl, 'warning', '⚠', 'Network error during verification', null, false);
    btn.disabled = false;
    statusEl.textContent = 'Error — retry';
  }
}

function renderResult(el, type, icon, name, conf, final) {
  el.className = `result-panel ${type}`;
  el.innerHTML = `
    <span class="result-icon">${icon}</span>
    <div class="result-name">${name}</div>
    ${conf ? `<div class="result-conf">${conf}</div>` : ''}
    ${final !== false
      ? `<button class="btn-primary" onclick="showPage('landing')" style="max-width:220px;margin:0 auto">
           Back to Home
         </button>`
      : `<button class="btn-primary" onclick="retryVerify()" style="max-width:220px;margin:0 auto">
           Try Again
         </button>`
    }
  `;
  el.classList.remove('hidden');
}

function retryVerify() {
  document.getElementById('verify-result').classList.add('hidden');
  document.getElementById('btn-verify').disabled = false;
  document.getElementById('verify-status-msg').innerHTML =
    '<span class="pulse-dot"></span> Camera ready — click Verify';
}


/* ─────────────────────────────────────────────────────────────
   UTILITIES
   ───────────────────────────────────────────────────────────── */
function showError(el, msg) {
  el.textContent = msg;
  el.classList.remove('hidden');
}

function setButtonLoading(btn, loading) {
  const textEl    = btn.querySelector('.btn-text');
  const spinnerEl = btn.querySelector('.btn-spinner');
  btn.disabled    = loading;
  if (textEl)    textEl.classList.toggle('hidden', loading);
  if (spinnerEl) spinnerEl.classList.toggle('hidden', !loading);
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


/* ─────────────────────────────────────────────────────────────
   BOOTSTRAP
   ───────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  showPage('landing');
});

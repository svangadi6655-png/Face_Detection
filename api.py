"""
api.py  —  FastAPI backend for the Face Recognition Portal
Run with:  uvicorn api:app --reload --port 8000
"""
import base64
import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr

import db
from model_manager import manager as model_manager

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Face Recognition Portal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
db.init_db()

# Start model training on startup (non-blocking)
model_manager.train(blocking=False)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class CaptureRequest(BaseModel):
    email: str
    frames: list[str]   # list of base64-encoded JPEG frames

class VerifyRequest(BaseModel):
    frame: str          # single base64-encoded JPEG frame


# ── Helper ────────────────────────────────────────────────────────────────────
def decode_b64_frame(b64_string: str) -> np.ndarray:
    """Decode a base64 data-URL or raw base64 JPEG into a BGR numpy array."""
    # Strip data URL header if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def get_session_user(authorization: str | None) -> dict:
    """Extract and validate Bearer token, return session dict."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    session = db.get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    return session


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/register")
def register(req: RegisterRequest):
    """Step 1 of registration: create account credentials."""
    full_name = req.full_name.strip()
    if not full_name:
        raise HTTPException(status_code=400, detail="Full name is required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    try:
        user = db.create_user(full_name, req.email, req.password)
        return {"success": True, "message": f"Account created for {full_name}.", "full_name": full_name}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/capture-faces")
def capture_faces(req: CaptureRequest):
    """Step 2 of registration: receive face frames and save them to disk."""
    user = db.get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    face_folder = user["face_folder"]
    os.makedirs(face_folder, exist_ok=True)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    saved = 0
    # Start numbering after any existing images
    existing = [f for f in os.listdir(face_folder) if f.lower().endswith(".jpg")]
    offset = len(existing)

    for b64 in req.frames:
        try:
            frame = decode_b64_frame(b64)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
            if len(faces) == 0:
                continue
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, (100, 100))
            img_path = os.path.join(face_folder, f"face_{offset + saved:04d}.jpg")
            cv2.imwrite(img_path, face_resized)
            saved += 1
        except Exception as e:
            print(f"[capture-faces] Error processing frame: {e}")
            continue

    if saved == 0:
        raise HTTPException(
            status_code=422,
            detail="No faces detected in any of the provided frames. Please retry in good lighting."
        )

    # Update face count in DB
    total_count = offset + saved
    db.update_face_count(req.email, total_count)

    # Trigger background model retrain
    model_manager.train(blocking=False)

    return {
        "success": True,
        "saved": saved,
        "total": total_count,
        "message": f"Captured {saved} face images. Model is retraining in the background."
    }


@app.post("/api/login")
def login(req: LoginRequest):
    """Verify credentials and return a session token."""
    user = db.verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if user["face_images_count"] == 0:
        raise HTTPException(
            status_code=403,
            detail="No face images registered for this account. Please complete registration."
        )
    token = db.create_session(user["id"], user["email"], user["full_name"])
    return {
        "success": True,
        "token": token,
        "full_name": user["full_name"],
        "email": user["email"]
    }


@app.post("/api/verify-face")
def verify_face(req: VerifyRequest, authorization: str = Header(default=None)):
    """Live face verification after login credentials are confirmed."""
    session = get_session_user(authorization)

    if model_manager.status == "training":
        raise HTTPException(status_code=503, detail="Model is currently retraining. Please wait a moment.")
    if model_manager.status != "ready":
        raise HTTPException(status_code=503, detail=f"Model not ready: {model_manager.status}. {model_manager.error_message}")

    try:
        frame = decode_b64_frame(req.frame)
        if frame is None:
            raise HTTPException(status_code=422, detail="Could not decode the image frame.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Frame decoding error: {e}")

    result = model_manager.predict(frame)

    # Enrich result with the logged-in user context
    result["logged_in_as"] = session["full_name"]
    result["match"] = (result["name"].lower() == session["full_name"].lower())

    # Logout after verification attempt (one-time use session)
    db.delete_session(session["token"])

    return result


@app.get("/api/model-status")
def model_status():
    return {
        "status": model_manager.status,
        "error": model_manager.error_message,
        "classes": list(model_manager._name_to_label.keys()) if model_manager.status == "ready" else []
    }


@app.post("/api/logout")
def logout(authorization: str = Header(default=None)):
    session = get_session_user(authorization)
    db.delete_session(session["token"])
    return {"success": True}


# ── Serve Frontend ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/{full_path:path}")
def catch_all(full_path: str):
    # SPA fallback
    return FileResponse("frontend/index.html")

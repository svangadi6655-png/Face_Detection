import sqlite3
import hashlib
import os
import secrets
from datetime import datetime

DB_PATH = "users.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            face_folder TEXT NOT NULL,
            face_images_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            email TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(full_name: str, email: str, password: str) -> dict:
    """Create a new user. Returns user dict or raises ValueError on duplicate email."""
    conn = get_connection()
    try:
        face_folder = f"dataset/dataset/faces/{full_name}"
        os.makedirs(face_folder, exist_ok=True)
        pw_hash = hash_password(password)
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO users (full_name, email, password_hash, face_folder, created_at) VALUES (?,?,?,?,?)",
            (full_name, email, pw_hash, face_folder, now)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row)
    except sqlite3.IntegrityError:
        raise ValueError("Email already registered.")
    finally:
        conn.close()

def verify_user(email: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict or None."""
    conn = get_connection()
    try:
        pw_hash = hash_password(password)
        row = conn.execute(
            "SELECT * FROM users WHERE email=? AND password_hash=?",
            (email, pw_hash)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_user_by_email(email: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def update_face_count(email: str, count: int):
    conn = get_connection()
    try:
        conn.execute("UPDATE users SET face_images_count=? WHERE email=?", (count, email))
        conn.commit()
    finally:
        conn.close()

def create_session(user_id: int, email: str, full_name: str) -> str:
    token = secrets.token_hex(32)
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (token, user_id, email, full_name, created_at) VALUES (?,?,?,?,?)",
            (token, user_id, email, full_name, now)
        )
        conn.commit()
        return token
    finally:
        conn.close()

def get_session(token: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def delete_session(token: str):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        conn.commit()
    finally:
        conn.close()

def get_all_enrolled_users() -> list[dict]:
    """Returns all users with at least some face images."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT full_name, face_folder FROM users WHERE face_images_count > 0").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

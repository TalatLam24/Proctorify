"""
Proctorify Live Proctor Server
==============================
Main server for the live proctoring system. Handles:
  - Serving HTML pages (login, register, lobby, exam, result)
  - Authentication (login, registration with bcrypt password hashing)
  - Exam and session management (create, submit, summary)
  - Real-time proctoring via WebSocket (10 fps frame streaming)
  - Feature extraction (reuses streaming_feature_extractor.py)
  - Model prediction (reuses predictor.py)
  - Alert saving to database
  - Debug frame saving when alerts fire (Option B)

Usage:
    python server.py

The server runs at http://127.0.0.1:8000
"""

import os
import sys
import json
import time
import uuid
import sqlite3
import base64
import asyncio
import bcrypt
import yaml
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
from yolo_detector import YoloPhoneDetector

# FastAPI and WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from streaming_feature_extractor import StreamingFeatureExtractor
from predictor import Predictor


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config():
    config_path = os.path.join(SCRIPT_DIR, 'server_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Resolve relative paths
def resolve_path(relative_path):
    return os.path.abspath(os.path.join(SCRIPT_DIR, relative_path))

DB_PATH = resolve_path(CONFIG['database']['path'])
FRONTEND_DIR = resolve_path(CONFIG['server']['frontend_dir'])
MODEL_PATH = resolve_path(CONFIG['model']['path'])
FACE_MODEL_PATH = resolve_path(CONFIG['features']['face_model_path'])
DEBUG_FRAMES_DIR = resolve_path(CONFIG['debug']['debug_frames_dir'])
YOLO_MODEL_PATH = resolve_path(CONFIG['yolo']['model_path'])


# ============================================================================
# DATABASE HELPER
# ============================================================================

def get_db():
    """Get a SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    return conn


# ============================================================================
# SESSION TOKEN MANAGEMENT (simple in-memory store)
# ============================================================================

# Maps token -> {user_id, email, name, expires_at}
active_tokens = {}

def create_session_token(user_id, email, name):
    """Create a session token for a logged-in user."""
    token = str(uuid.uuid4())
    active_tokens[token] = {
        'user_id': user_id,
        'email': email,
        'name': name,
        'expires_at': time.time() + 3600 * 24  # 24 hours
    }
    return token

def validate_session_token(token):
    """Validate a session token. Returns user info or None."""
    if not token:
        return None
    session = active_tokens.get(token)
    if not session:
        return None
    if time.time() > session['expires_at']:
        del active_tokens[token]
        return None
    return session

def get_current_user(request: Request):
    """Extract and validate the session token from the request."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
    else:
        token = request.cookies.get('session_token', '')
    return validate_session_token(token)


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(title="Proctorify Live Proctor")

# CORS (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================================
# HTML PAGE ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the login page."""
    with open(os.path.join(FRONTEND_DIR, 'index.html'), 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    """Serve the registration page."""
    with open(os.path.join(FRONTEND_DIR, 'register.html'), 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/lobby", response_class=HTMLResponse)
async def lobby_page():
    """Serve the lobby page."""
    with open(os.path.join(FRONTEND_DIR, 'lobby.html'), 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/exam", response_class=HTMLResponse)
async def exam_page():
    """Serve the exam page."""
    with open(os.path.join(FRONTEND_DIR, 'exam.html'), 'r', encoding='utf-8') as f:
        return f.read()

@app.get("/result", response_class=HTMLResponse)
async def result_page():
    """Serve the result page."""
    with open(os.path.join(FRONTEND_DIR, 'result.html'), 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================================
# AUTHENTICATION API
# ============================================================================

@app.post("/api/login")
async def login(request: Request):
    """Handle login form submission."""
    data = await request.json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return JSONResponse({'error': 'Email and password are required'}, status_code=400)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, password_hash FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return JSONResponse({'error': 'Invalid email or password'}, status_code=401)

    # Verify password
    if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return JSONResponse({'error': 'Invalid email or password'}, status_code=401)

    # Create session token
    token = create_session_token(user['id'], email, user['name'])

    # Update last_login
    conn = get_db()
    conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user['id']))
    conn.commit()
    conn.close()

    response = JSONResponse({'token': token, 'name': user['name']})
    response.set_cookie('session_token', token, httponly=True, max_age=3600*24)
    return response

@app.post("/api/register")
async def register(request: Request):
    """Handle registration form submission."""
    data = await request.json()
    email = data.get('email', '').strip().lower()
    name = data.get('name', '').strip()
    password = data.get('password', '')

    if not email or not name or not password:
        return JSONResponse({'error': 'All fields are required'}, status_code=400)
    if len(password) < 6:
        return JSONResponse({'error': 'Password must be at least 6 characters'}, status_code=400)

    conn = get_db()
    cursor = conn.cursor()

    # Check if email already exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return JSONResponse({'error': 'Email already registered'}, status_code=409)

    # Hash password and insert
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cursor.execute(
        "INSERT INTO users (email, name, password_hash, role) VALUES (?, ?, ?, 'student')",
        (email, name, password_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    # Create session token
    token = create_session_token(user_id, email, name)

    response = JSONResponse({'token': token, 'name': name})
    response.set_cookie('session_token', token, httponly=True, max_age=3600*24)
    return response

@app.post("/api/logout")
async def logout(request: Request):
    """Handle logout."""
    user = get_current_user(request)
    if user:
        # Find and remove the token
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
        else:
            token = request.cookies.get('session_token', '')
        if token in active_tokens:
            del active_tokens[token]

    response = JSONResponse({'success': True})
    response.delete_cookie('session_token')
    return response

@app.get("/api/me")
async def get_me(request: Request):
    """Get current user info."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({'error': 'Not authenticated'}, status_code=401)
    return JSONResponse({'name': user['name'], 'email': user['email']})


# ============================================================================
# EXAM AND SESSION API
# ============================================================================

@app.get("/api/exams")
async def get_exams(request: Request):
    """Get list of active exams."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({'error': 'Not authenticated'}, status_code=401)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, description, duration_minutes, questions FROM exams WHERE is_active = 1 ORDER BY id"
    )
    exams = cursor.fetchall()
    conn.close()

    result = []
    for exam in exams:
        result.append({
            'id': exam['id'],
            'title': exam['title'],
            'description': exam['description'],
            'duration_minutes': exam['duration_minutes'],
            'question_count': len(json.loads(exam['questions']))
        })
    return JSONResponse({'exams': result})

@app.post("/api/session/start")
async def start_session(request: Request):
    """Start a new exam session."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({'error': 'Not authenticated'}, status_code=401)

    data = await request.json()
    exam_id = data.get('exam_id')

    conn = get_db()
    cursor = conn.cursor()

    # Get exam details
    cursor.execute("SELECT id, title, duration_minutes, questions FROM exams WHERE id = ? AND is_active = 1", (exam_id,))
    exam = cursor.fetchone()
    if not exam:
        conn.close()
        return JSONResponse({'error': 'Exam not found or inactive'}, status_code=404)

    # Create new session
    session_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO sessions (id, user_id, exam_id, started_at, status) VALUES (?, ?, ?, ?, 'active')",
        (session_id, user['user_id'], exam_id, datetime.now())
    )
    conn.commit()
    conn.close()

    return JSONResponse({
        'session_id': session_id,
        'exam': {
            'id': exam['id'],
            'title': exam['title'],
            'duration_minutes': exam['duration_minutes'],
            'questions': json.loads(exam['questions'])
        }
    })

@app.post("/api/session/{session_id}/submit")
async def submit_session(session_id: str, request: Request):
    """Submit exam answers and mark session as completed."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({'error': 'Not authenticated'}, status_code=401)

    data = await request.json()
    answers = data.get('answers', {})

    conn = get_db()
    cursor = conn.cursor()

    # Verify session belongs to this user
    cursor.execute("SELECT id, user_id, exam_id FROM sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    if not session:
        conn.close()
        return JSONResponse({'error': 'Session not found'}, status_code=404)
    if session['user_id'] != user['user_id']:
        conn.close()
        return JSONResponse({'error': 'Unauthorized'}, status_code=403)

    # Save answers
    for question_index, answer_text in answers.items():
        cursor.execute(
            "INSERT INTO exam_answers (session_id, question_index, answer_text) VALUES (?, ?, ?)",
            (session_id, int(question_index), answer_text)
        )

    # Mark session as completed
    cursor.execute(
        "UPDATE sessions SET ended_at = ?, status = 'completed' WHERE id = ?",
        (datetime.now(), session_id)
    )
    conn.commit()

    # Get session summary
    cursor.execute("""
        SELECT s.total_frames, s.flagged_frames, s.flag_percentage,
               s.alert_count, s.started_at, s.ended_at,
               e.title, e.duration_minutes
        FROM sessions s
        JOIN exams e ON s.exam_id = e.id
        WHERE s.id = ?
    """, (session_id,))
    summary = cursor.fetchone()
    conn.close()

    # Count answered questions
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM exam_answers WHERE session_id = ?", (session_id,))
    answered_count = cursor.fetchone()['count']
    conn.close()

    return JSONResponse({
        'session_id': session_id,
        'exam_title': summary['title'],
        'total_frames': summary['total_frames'],
        'flagged_frames': summary['flagged_frames'],
        'flag_percentage': summary['flag_percentage'],
        'alert_count': summary['alert_count'],
        'answered_count': answered_count,
        'started_at': str(summary['started_at']),
        'ended_at': str(summary['ended_at'])
    })

@app.get("/api/session/{session_id}/summary")
async def get_session_summary(session_id: str, request: Request):
    """Get a session's summary (for the result page)."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({'error': 'Not authenticated'}, status_code=401)

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.total_frames, s.flagged_frames, s.flag_percentage,
               s.alert_count, s.started_at, s.ended_at, s.status,
               e.title, e.duration_minutes
        FROM sessions s
        JOIN exams e ON s.exam_id = e.id
        WHERE s.id = ? AND s.user_id = ?
    """, (session_id, user['user_id']))
    summary = cursor.fetchone()

    if not summary:
        conn.close()
        return JSONResponse({'error': 'Session not found'}, status_code=404)

    cursor.execute("SELECT COUNT(*) as count FROM exam_answers WHERE session_id = ?", (session_id,))
    answered_count = cursor.fetchone()['count']
    conn.close()

    return JSONResponse({
        'session_id': session_id,
        'exam_title': summary['title'],
        'duration_minutes': summary['duration_minutes'],
        'total_frames': summary['total_frames'],
        'flagged_frames': summary['flagged_frames'],
        'flag_percentage': summary['flag_percentage'] or 0.0,
        'alert_count': summary['alert_count'] or 0,
        'answered_count': answered_count,
        'status': summary['status'],
        'started_at': str(summary['started_at']),
        'ended_at': str(summary['ended_at']) if summary['ended_at'] else None
    })


# ============================================================================
# WEBSOCKET: REAL-TIME PROCTORING
# ============================================================================

# Store active WebSocket sessions
# Maps session_id -> {websocket, extractor, predictor, user_id, alert_count, debug_dir}
ws_sessions = {}

@app.websocket("/ws/proctor/{session_id}")
async def proctor_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time proctoring.

    Architecture: Two concurrent loops
      - receive_loop: receives all messages, puts frames in a bounded queue
      - process_loop: processes frames from the queue in a thread

    Frame dropping: if the queue is full (server is behind), the oldest
    frame is dropped. This ensures we always process the latest available
    frame, eliminating lag.
    """
    await websocket.accept()

    # Verify session exists
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_id FROM sessions WHERE id = ? AND status = 'active'", (session_id,))
    session = cursor.fetchone()
    conn.close()

    if not session:
        await websocket.send_json({'type': 'error', 'message': 'Invalid or inactive session'})
        await websocket.close()
        return

    # Initialize feature extractor and predictor
    try:
        extractor = StreamingFeatureExtractor(
            face_model_path=FACE_MODEL_PATH,
            variance_window_sec=CONFIG['features']['variance_window_sec'],
            max_event_buffer_age=CONFIG['features']['max_event_buffer_age_sec']
        )
    except FileNotFoundError as e:
        await websocket.send_json({'type': 'error', 'message': str(e)})
        await websocket.close()
        return

    predictor = Predictor(
        model_path=MODEL_PATH,
        threshold=CONFIG['model']['threshold'],
        ema_alpha=CONFIG['model']['ema_alpha']
    )

    # Debug frame saving setup
    debug_save = CONFIG['debug']['save_alert_frames']
    debug_dir = None
    if debug_save:
        debug_dir = os.path.join(DEBUG_FRAMES_DIR, session_id)
        os.makedirs(debug_dir, exist_ok=True)

    # Register this WebSocket session
    ws_sessions[session_id] = {
        'websocket': websocket,
        'extractor': extractor,
        'predictor': predictor,
        'user_id': session['user_id'],
        'alert_count': 0,
        'debug_dir': debug_dir,
        'last_db_update': time.time()
    }

    print(f"[WS] Session {session_id[:8]}... connected")

    # Bounded queue: only keep the latest frame
    frame_queue = asyncio.Queue(maxsize=1)

    async def receive_loop():
        """Receive messages. Frames go to queue, events handled immediately."""
        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get('type')

                if msg_type == 'start':
                    await websocket.send_json({'type': 'ready'})

                elif msg_type == 'frame':
                    # Drop old frame if queue is full, then add new frame
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(message)

                elif msg_type == 'keyboard_event':
                    timestamp = message.get('timestamp', time.time())
                    key = message.get('key', '')
                    extractor.add_keyboard_event(timestamp, key)

                elif msg_type == 'mouse_event':
                    timestamp = message.get('timestamp', time.time())
                    extractor.add_mouse_event(timestamp, message.get('button', ''))

                elif msg_type == 'digital_cheat_event':
                    # Handle immediately (rare event, OK to block briefly)
                    timestamp = message.get('timestamp', time.time())
                    reason = message.get('reason', 'unknown')

                    conn = get_db()
                    conn.execute(
                        """INSERT INTO alerts
                           (session_id, started_at, ended_at, duration_seconds,
                            peak_probability, avg_probability, label, category)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (session_id, datetime.fromtimestamp(timestamp),
                         datetime.fromtimestamp(timestamp), 0.0, 1.0, 1.0,
                         'CHEAT_DIGITAL', reason)
                    )
                    ws_sessions[session_id]['alert_count'] += 1
                    conn.execute(
                        "UPDATE sessions SET alert_count = ? WHERE id = ?",
                        (ws_sessions[session_id]['alert_count'], session_id)
                    )
                    conn.commit()
                    conn.close()

                    await websocket.send_json({
                        'type': 'digital_cheat_confirmed',
                        'reason': reason,
                        'alert_number': ws_sessions[session_id]['alert_count']
                    })
                    print(f"[WS] CHEAT_DIGITAL detected: {reason} (session {session_id[:8]}...)")

                elif msg_type == 'end':
                    await frame_queue.put({'type': 'end'})
                    break

        except WebSocketDisconnect:
            print(f"[WS] Session {session_id[:8]}... disconnected")
        except Exception as e:
            print(f"[WS] Receive error: {e}")
        finally:
            # Signal process loop to stop
            try:
                frame_queue.put_nowait({'type': 'end'})
            except asyncio.QueueFull:
                pass

    async def process_loop():
        """Process frames from queue in a thread (non-blocking)."""
        try:
            while True:
                message = await frame_queue.get()
                if message.get('type') == 'end':
                    break

                # Process frame in a thread (doesn't block the event loop)
                # Initialize YOLO detector if not already created for this session
                if 'yolo_detector' not in ws_sessions[session_id]:
                    try:
                        ws_sessions[session_id]['yolo_detector'] = YoloPhoneDetector(
                            model_path=YOLO_MODEL_PATH,
                            confidence_threshold=CONFIG['yolo']['confidence_threshold'],
                            target_class=CONFIG['yolo']['target_class']
                        )
                    except Exception as e:
                        print(f"[WS] YOLO init failed: {e}")
                        ws_sessions[session_id]['yolo_detector'] = None
                
                messages_to_send = await asyncio.to_thread(
                    process_frame_sync,
                    message, session_id, extractor, predictor, debug_dir, debug_save,
                    ws_sessions[session_id].get('yolo_detector')
                )

                # Send all messages
                for msg in messages_to_send:
                    await websocket.send_json(msg)

                # Periodic DB update (every 5 seconds)
                current_time = time.time()
                if current_time - ws_sessions[session_id]['last_db_update'] > 5.0:
                    stats = predictor.get_session_stats()
                    conn = get_db()
                    conn.execute(
                        """UPDATE sessions SET
                           total_frames = ?, flagged_frames = ?,
                           flag_percentage = ?, avg_probability = ?,
                           max_probability = ?
                           WHERE id = ?""",
                        (stats['total_frames'], stats['flagged_frames'],
                         stats['flag_percentage'], stats['avg_probability'],
                         stats['max_probability'], session_id)
                    )
                    conn.commit()
                    conn.close()
                    ws_sessions[session_id]['last_db_update'] = current_time

        except Exception as e:
            print(f"[WS] Process error: {e}")

    # Run both loops concurrently
    await asyncio.gather(receive_loop(), process_loop())

    # Final DB update
    stats = predictor.get_session_stats()
    conn = get_db()
    conn.execute(
        """UPDATE sessions SET
           total_frames = ?, flagged_frames = ?,
           flag_percentage = ?, avg_probability = ?,
           max_probability = ?, alert_count = ?
           WHERE id = ?""",
        (stats['total_frames'], stats['flagged_frames'],
         stats['flag_percentage'], stats['avg_probability'],
         stats['max_probability'], ws_sessions[session_id]['alert_count'],
         session_id)
    )
    conn.commit()
    conn.close()

    # Cleanup
    extractor.close()
    if session_id in ws_sessions:
        del ws_sessions[session_id]
    print(f"[WS] Session {session_id[:8]}... cleaned up")


def process_frame_sync(message, session_id, extractor, predictor, debug_dir, debug_save, yolo_detector):
    """
    Process a single frame synchronously (runs in a thread).
    Returns list of messages to send back to the client.
    """
    messages_to_send = []

    frame_data = message.get('data', '')
    timestamp = message.get('timestamp', time.time())

    try:
        # Decode base64 JPEG
        img_bytes = base64.b64decode(frame_data.split(',')[-1] if ',' in frame_data else frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return messages_to_send

        # Extract features
        features, face_detected = extractor.process_frame(frame, timestamp)

        # --- YOLO Phone Detection ---
        phone_detected = False
        phone_confidence = 0.0
        if yolo_detector is not None:
            phone_detected, phone_confidence = yolo_detector.detect(frame)
        
        # --- Run ML Prediction (Force suspicious if YOLO detects phone) ---
        prob, label, alert_event, alert_info = predictor.predict(
            features, timestamp, force_suspicious=phone_detected
        )
        
        # --- YOLO Notification ---
        if phone_detected:
            messages_to_send.append({
                'type': 'yolo_detection',
                'phone_detected': True,
                'confidence': round(phone_confidence, 2)
            })

        # Build prediction message
        messages_to_send.append({
            'type': 'prediction',
            'probability': round(prob, 4),
            'label': label,
            'face_detected': face_detected,
            'frame_number': predictor.total_frames
        })

        # Handle alert events
        if alert_event == 'started':
            ws_sessions[session_id]['alert_count'] += 1
            messages_to_send.append({
                'type': 'alert',
                'event': 'started',
                'label': 'CHEAT_PHYSICAL',
                'alert_number': ws_sessions[session_id]['alert_count']
            })

        elif alert_event == 'ended':
            # Save alert to database
            conn = get_db()
            conn.execute(
                """INSERT INTO alerts
                   (session_id, started_at, ended_at, duration_seconds,
                    peak_probability, avg_probability, label)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id,
                 datetime.fromtimestamp(alert_info['started_at']),
                 datetime.fromtimestamp(alert_info['ended_at']),
                 alert_info['duration_seconds'],
                 alert_info['peak_probability'],
                 alert_info['avg_probability'],
                 alert_info['label'])
            )
            conn.execute(
                "UPDATE sessions SET alert_count = ? WHERE id = ?",
                (ws_sessions[session_id]['alert_count'], session_id)
            )
            conn.commit()
            conn.close()

            messages_to_send.append({
                'type': 'alert',
                'event': 'ended',
                'duration': round(alert_info['duration_seconds'], 2),
                'peak_probability': round(alert_info['peak_probability'], 4),
                'label': 'CHEAT_PHYSICAL'
            })

        # Save debug frame if alert is active
        if debug_save and debug_dir and predictor.in_alert:
            frame_filename = f"{predictor.total_frames:06d}_{prob:.4f}.jpg"
            frame_path = os.path.join(debug_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

    except Exception as e:
        print(f"[WS] Frame processing error: {e}")
        messages_to_send.append({'type': 'error', 'message': f'Frame error: {str(e)}'})

    return messages_to_send


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROCTORIFY LIVE PROCTOR SERVER")
    print("=" * 60)
    print(f"Database:     {DB_PATH}")
    print(f"Frontend:     {FRONTEND_DIR}")
    print(f"Model:        {MODEL_PATH}")
    print(f"Face model:   {FACE_MODEL_PATH}")
    print(f"Debug frames: {DEBUG_FRAMES_DIR}")
    print(f"Threshold:    {CONFIG['model']['threshold']}")
    print(f"EMA alpha:    {CONFIG['model']['ema_alpha']}")
    print(f"Frame rate:   {CONFIG['streaming']['frame_rate']} fps")
    print(f"JPEG quality: {CONFIG['streaming']['jpeg_quality']}")
    print()

    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Make sure you've trained the model in Approach 03.")
        sys.exit(1)

    # Verify face model exists
    if not os.path.exists(FACE_MODEL_PATH):
        print(f"ERROR: Face model not found at {FACE_MODEL_PATH}")
        print("Download face_landmarker.task from MediaPipe.")
        sys.exit(1)

    # Verify database exists
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run init_db.py first.")
        sys.exit(1)

    print("Starting server...")
    print(f"Open http://{CONFIG['server']['host']}:{CONFIG['server']['port']} in your browser")
    print()

    uvicorn.run(
        app,
        host=CONFIG['server']['host'],
        port=CONFIG['server']['port']
    )
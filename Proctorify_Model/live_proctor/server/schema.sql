-- Proctorify Live Proctor Database Schema
-- Version 1.0
-- 
-- This schema defines 5 tables:
--   1. users          - Student/instructor accounts for authentication
--   2. exams          - Exam content (title, questions, duration)
--   3. sessions       - One row per exam attempt
--   4. alerts         - Sustained suspicious periods during a session
--   5. exam_answers   - Student's submitted answers
--
-- The schema is created by init_db.py, which also inserts:
--   - One demo exam (5 TCP/UDP questions)
--   - (Users are created via the registration page, not in init_db.py)

-- ============================================================================
-- Table 1: users
-- ============================================================================
-- Stores user accounts for authentication.
-- Password is hashed with bcrypt (never stored in plaintext).
-- The 'role' field supports future expansion to instructor dashboard.

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'student',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Index for fast login lookup
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ============================================================================
-- Table 2: exams
-- ============================================================================
-- Stores exam content. Each exam has a list of questions stored as JSON.
-- Questions are stored as a JSON array of strings, e.g.:
--   ["Question 1 text", "Question 2 text", "Question 3 text"]
--
-- Only exams with is_active=1 are shown in the lobby.

CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    duration_minutes INTEGER NOT NULL,
    questions TEXT NOT NULL,  -- JSON array of question strings
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast active-exam lookup
CREATE INDEX IF NOT EXISTS idx_exams_active ON exams(is_active);

-- ============================================================================
-- Table 3: sessions
-- ============================================================================
-- One row per exam attempt. Created when student clicks "Start Exam",
-- updated continuously during the exam, marked as completed on submit.
--
-- The 'id' is a UUID (text) for security — students can't guess other
-- session IDs by incrementing integers.

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,  -- UUID
    user_id INTEGER NOT NULL,
    exam_id INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT DEFAULT 'active',  -- 'active', 'completed', 'aborted'
    total_frames INTEGER DEFAULT 0,
    flagged_frames INTEGER DEFAULT 0,
    flag_percentage REAL DEFAULT 0.0,
    avg_probability REAL DEFAULT 0.0,
    max_probability REAL DEFAULT 0.0,
    alert_count INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (exam_id) REFERENCES exams(id)
);

-- Index for fast "user's past sessions" lookup
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, started_at);

-- Index for fast "active session" lookup
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

-- ============================================================================
-- Table 4: alerts
-- ============================================================================
-- Stores sustained suspicious periods during a session.
-- An alert represents a continuous period where the model's probability
-- was above threshold, NOT individual frame predictions.
--
-- Example: A student looks at their phone for 5 seconds.
-- The model's probability rises above 0.06 at t=10s, stays above until t=15s.
-- This generates ONE alert with started_at=10s, ended_at=15s, duration=5s.

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    duration_seconds REAL,
    peak_probability REAL NOT NULL,
    avg_probability REAL NOT NULL,
    label TEXT NOT NULL,  -- 'CHEAT_PHYSICAL' or 'CHEAT_DIGITAL'
    category TEXT,  -- Optional: e.g., 'head_turned', 'gaze_away'. NULL for now.
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Index for fast "alerts in session" lookup
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id, started_at);

-- ============================================================================
-- Table 5: exam_answers
-- ============================================================================
-- Stores the student's submitted answers. All answers are saved at once
-- when the student clicks "Submit Exam" (not incrementally as they type).

CREATE TABLE IF NOT EXISTS exam_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question_index INTEGER NOT NULL,  -- 0, 1, 2, ... (position in questions JSON)
    answer_text TEXT NOT NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Index for fast "answers in session" lookup
CREATE INDEX IF NOT EXISTS idx_answers_session ON exam_answers(session_id);
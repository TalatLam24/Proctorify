"""
Proctorify Live Proctor - Database Initialization Script
=======================================================
Creates the SQLite database with the 5 tables defined in schema.sql.
Inserts one demo exam (5 TCP/UDP questions).

Usage:
    python init_db.py

This script:
  1. Creates the database file at ../data/proctorify.db (if it doesn't exist)
  2. Creates all 5 tables with indexes
  3. Inserts one demo exam if no exams exist

The script is idempotent — running it multiple times is safe.
Existing data is preserved. Only missing tables/exams are created.

NOTE: This script does NOT create user accounts.
Users are created via the registration page in the web interface.
"""

import os
import sys
import sqlite3
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR.parent / "data"
DB_PATH = DATA_DIR / "proctorify.db"
SCHEMA_PATH = SCRIPT_DIR / "schema.sql"

# Demo exam content (the TCP/UDP questions from the original exam.html)
DEMO_EXAM = {
    "title": "CSE 4098B Practice Exam",
    "description": "A practice exam covering networking, algorithms, OS, databases, and concurrency.",
    "duration_minutes": 10,
    "questions": [
        "Explain the difference between TCP and UDP, and describe one real-world scenario where UDP is the better choice.",
        "What is the time complexity of binary search, and why does the array need to be sorted beforehand?",
        "Describe the difference between a process and a thread, and give one advantage of using threads.",
        "What is normalization in databases? Briefly explain 1NF and 2NF with an example.",
        "Explain what a deadlock is in operating systems and list two ways to prevent it."
    ]
}


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_database():
    """Create the database, schema, and seed data."""
    print("=" * 60)
    print("PROCTORIFY LIVE PROCTOR - DATABASE INITIALIZATION")
    print("=" * 60)
    print(f"Database path: {DB_PATH}")
    print(f"Schema path:   {SCHEMA_PATH}")
    print()

    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory ready: {DATA_DIR}")

    # Check if schema file exists
    if not SCHEMA_PATH.exists():
        print(f"ERROR: Schema file not found at {SCHEMA_PATH}")
        print("Please make sure schema.sql is in the same folder as init_db.py.")
        sys.exit(1)

    # Connect to (or create) the database
    print(f"Connecting to database...")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Read and execute the schema
    print(f"Reading schema from {SCHEMA_PATH.name}...")
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    print(f"Schema executed successfully.")

    # Check if any exams exist
    cursor.execute("SELECT COUNT(*) FROM exams")
    exam_count = cursor.fetchone()[0]

    if exam_count == 0:
        print(f"No exams found. Inserting demo exam...")
        import json
        cursor.execute(
            """INSERT INTO exams (title, description, duration_minutes, questions, is_active)
               VALUES (?, ?, ?, ?, 1)""",
            (
                DEMO_EXAM["title"],
                DEMO_EXAM["description"],
                DEMO_EXAM["duration_minutes"],
                json.dumps(DEMO_EXAM["questions"])
            )
        )
        conn.commit()
        print(f"Demo exam inserted: '{DEMO_EXAM['title']}' with {len(DEMO_EXAM['questions'])} questions.")
    else:
        print(f"Found {exam_count} existing exam(s). Skipping demo exam insertion.")

    # Show summary
    print()
    print("=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    # Show tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    print(f"Tables ({len(tables)}):")
    for (table_name,) in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  - {table_name}: {count} rows")

    # Show exams
    print()
    cursor.execute("SELECT id, title, is_active FROM exams ORDER BY id")
    exams = cursor.fetchall()
    print(f"Exams ({len(exams)}):")
    for exam_id, title, is_active in exams:
        status = "ACTIVE" if is_active else "INACTIVE"
        print(f"  [{exam_id}] {title} ({status})")

    # Show users
    print()
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    print(f"Users: {user_count}")
    if user_count == 0:
        print(f"  (No users yet. Create one via the registration page.)")

    conn.close()
    print()
    print("Database initialization complete.")
    print(f"Database file: {DB_PATH}")


if __name__ == "__main__":
    init_database()
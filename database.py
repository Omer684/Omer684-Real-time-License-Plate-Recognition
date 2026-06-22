import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "plates.db")

def _get_connection():
    # check_same_thread=False allows sharing the connection among Flask threads
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = _get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            mode TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_detection(text: str, timestamp: str, date: str, mode: str):
    conn = _get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections (text, timestamp, date, mode)
        VALUES (?, ?, ?, ?)
    ''', (text, timestamp, date, mode))
    conn.commit()
    conn.close()

def get_recent_detections(limit: int = 50) -> list[dict]:
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT id, text, timestamp, date, mode
        FROM detections
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_total_detections() -> int:
    conn = _get_connection()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM detections')
    count = c.fetchone()[0]
    conn.close()
    return count

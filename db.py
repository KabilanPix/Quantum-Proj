import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from config import DB_PATH

def get_connection() -> sqlite3.Connection:
    """Creates a sqlite3 connection with WAL mode enabled and timeout."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes readings, events, risk_scores, schedules, and alerts tables."""
    with get_connection() as conn:
        # Readings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                temperature REAL NOT NULL,
                vibration REAL NOT NULL,
                power REAL NOT NULL,
                status TEXT NOT NULL,
                source TEXT DEFAULT 'simulator'
            );
        """)
        
        # Check if source column exists in readings (safe migration)
        cursor = conn.execute("PRAGMA table_info(readings);")
        cols = [row["name"] for row in cursor.fetchall()]
        if "source" not in cols:
            conn.execute("ALTER TABLE readings ADD COLUMN source TEXT DEFAULT 'simulator';")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_machine_time 
            ON readings (machine_id, timestamp);
        """)

        # Events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                type TEXT NOT NULL,
                detail TEXT
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_time 
            ON events (timestamp);
        """)

        # Risk scores table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                risk REAL NOT NULL,
                level TEXT NOT NULL,
                top_factor TEXT,
                anomaly_score REAL DEFAULT 0.0
            );
        """)

        # Schedules table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                assignment_json TEXT NOT NULL,
                cost REAL NOT NULL,
                solver TEXT NOT NULL,
                classical_optimal_cost REAL,
                risk_snapshot_json TEXT
            );
        """)

        # Alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                level TEXT NOT NULL,
                risk REAL NOT NULL,
                message TEXT NOT NULL,
                sent_telegram INTEGER DEFAULT 0
            );
        """)

        conn.commit()

def insert_reading(machine_id: str, timestamp: float, temperature: float, vibration: float, power: float, status: str, source: str = "simulator") -> int:
    """Inserts a new sensor reading into the database."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO readings (machine_id, timestamp, temperature, vibration, power, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (machine_id, timestamp, float(temperature), float(vibration), float(power), status, source)
        )
        conn.commit()
        return cursor.lastrowid

def insert_event(machine_id: str, timestamp: float, event_type: str, detail: str = "") -> int:
    """Inserts a maintenance or machine lifecycle event."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO events (machine_id, timestamp, type, detail)
            VALUES (?, ?, ?, ?)
            """,
            (machine_id, timestamp, event_type, detail)
        )
        conn.commit()
        return cursor.lastrowid

def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent events sorted descending by timestamp."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, machine_id, timestamp, type, detail
            FROM events
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]

def insert_risk_score(machine_id: str, timestamp: float, risk: float, level: str, top_factor: str = "", anomaly_score: float = 0.0) -> int:
    """Inserts an ML risk score and anomaly rating."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO risk_scores (machine_id, timestamp, risk, level, top_factor, anomaly_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (machine_id, timestamp, float(risk), level, top_factor, float(anomaly_score))
        )
        conn.commit()
        return cursor.lastrowid

def get_latest_risk_scores() -> List[Dict[str, Any]]:
    """Retrieves the latest risk score for each machine."""
    with get_connection() as conn:
        query = """
            SELECT r.machine_id, r.timestamp, r.risk, r.level, r.top_factor, r.anomaly_score
            FROM risk_scores r
            INNER JOIN (
                SELECT machine_id, MAX(timestamp) as max_time
                FROM risk_scores
                GROUP BY machine_id
            ) latest ON r.machine_id = latest.machine_id AND r.timestamp = latest.max_time;
        """
        cursor = conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]

def insert_schedule(timestamp: float, assignment_json: str, cost: float, solver: str, classical_optimal_cost: float, risk_snapshot_json: str) -> int:
    """Inserts an automated schedule record."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO schedules (timestamp, assignment_json, cost, solver, classical_optimal_cost, risk_snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, assignment_json, float(cost), solver, float(classical_optimal_cost), risk_snapshot_json)
        )
        conn.commit()
        return cursor.lastrowid

def get_latest_schedule() -> Optional[Dict[str, Any]]:
    """Retrieves the most recent automated schedule."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, timestamp, assignment_json, cost, solver, classical_optimal_cost, risk_snapshot_json
            FROM schedules
            ORDER BY timestamp DESC
            LIMIT 1;
            """
        )
        row = cursor.fetchone()
        return dict(row) if row else None

def get_schedule_history(limit: int = 30) -> List[Dict[str, Any]]:
    """Retrieves historical automated schedules."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, timestamp, assignment_json, cost, solver, classical_optimal_cost, risk_snapshot_json
            FROM schedules
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]

def insert_alert(machine_id: str, timestamp: float, level: str, risk: float, message: str, sent_telegram: int = 0) -> int:
    """Inserts a triggered alert record."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO alerts (machine_id, timestamp, level, risk, message, sent_telegram)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (machine_id, timestamp, level, float(risk), message, sent_telegram)
        )
        conn.commit()
        return cursor.lastrowid

def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent alerts."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, machine_id, timestamp, level, risk, message, sent_telegram
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]

def get_recent(machine_id: str, n: int = 200) -> pd.DataFrame:
    """Retrieves recent readings for a machine sorted by time ascending."""
    with get_connection() as conn:
        query = """
            SELECT machine_id, timestamp, temperature, vibration, power, status
            FROM (
                SELECT * FROM readings
                WHERE machine_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC;
        """
        df = pd.read_sql_query(query, conn, params=(machine_id, n))
        return df

def get_latest_all() -> List[Dict[str, Any]]:
    """Retrieves the latest reading per machine."""
    with get_connection() as conn:
        query = """
            SELECT r.machine_id, r.timestamp, r.temperature, r.vibration, r.power, r.status, r.source
            FROM readings r
            INNER JOIN (
                SELECT machine_id, MAX(timestamp) as max_time
                FROM readings
                GROUP BY machine_id
            ) latest ON r.machine_id = latest.machine_id AND r.timestamp = latest.max_time;
        """
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def clear_readings() -> None:
    """Clears all readings from database."""
    with get_connection() as conn:
        conn.execute("DELETE FROM readings;")
        conn.commit()

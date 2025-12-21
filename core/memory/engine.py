import json
import os
import sqlite3
from typing import Any, Dict, List


class MemoryEngine:
    """
    Personal Knowledge Graph Engine.
    Stores user preferences, playbooks, and insights.
    Persists to data/personal_memory.db by default.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Resolve to root/data/personal_memory.db
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base_dir, "data", "personal_memory.db")

        self.db_path = db_path
        self.conn = None

        # Keep connection open if memory, otherwise open/close on demand or keep open (sqlite handles concurrency poorly if multi-threaded, but here we are single threaded mostly)
        if self.db_path == ":memory:":
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self._init_db()

    def _get_conn(self):
        if self.db_path == ":memory:" and self.conn:
            return self.conn
        return sqlite3.connect(self.db_path)

    def _close_conn(self, conn):
        if self.db_path != ":memory:":
            conn.close()

    def _init_db(self):
        """Initialize SQLite storage."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                category TEXT,
                tags TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add FTS for keyword search if not exists?
        # Keeping it simple for now
        conn.commit()
        self._close_conn(conn)

    def store_memory(self, content: str, category: str, tags: List[str] = None):
        """Store a new memory item."""
        conn = self._get_conn()
        cursor = conn.cursor()
        tags_json = json.dumps(tags or [])
        cursor.execute(
            'INSERT INTO memories (content, category, tags) VALUES (?, ?, ?)',
            (content, category, tags_json)
        )
        conn.commit()
        self._close_conn(conn)

    def query_memory(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories.
        Uses simple LIKE matching as fallback for vector search.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Split query into keywords
        keywords = query_text.split()
        if not keywords:
            return []

        # Build query: content LIKE %k1% OR content LIKE %k2%
        conditions = []
        params = []
        for k in keywords:
            conditions.append("content LIKE ?")
            params.append(f"%{k}%")

        sql = f"SELECT content, category, tags, timestamp FROM memories WHERE {' OR '.join(conditions)} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        self._close_conn(conn)

        return [
            {
                "content": r[0],
                "category": r[1],
                "tags": json.loads(r[2]),
                "timestamp": r[3],
                "score": 0.9 # Mock score
            }
            for r in rows
        ]

    def get_context(self) -> str:
        """Constructs a context string for the LLM from recent/important memories."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content FROM memories WHERE category IN ('philosophy', 'core_beliefs') ORDER BY timestamp DESC LIMIT 10"
        )
        rows = cursor.fetchall()
        self._close_conn(conn)

        if not rows:
            return "No personal context available."

        context_lines = [f"- {r[0]}" for r in rows]
        return "\n".join(context_lines)

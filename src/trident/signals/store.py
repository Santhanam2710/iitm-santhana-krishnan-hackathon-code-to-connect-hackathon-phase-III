"""Signal persistence: SQLite (queryable, point-in-time) + append-only JSONL sink.

The JSONL file is the "write to a file" output the brief allows; the SQLite table backs
the REST API. Both are keyed on ``ts_published`` so backtests are point-in-time.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from trident.config import get_config, resolve
from trident.signals.schema import Signal

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY,
  ts_published TEXT NOT NULL,
  ts_emitted TEXT NOT NULL,
  ticker TEXT NOT NULL,
  event_label TEXT NOT NULL,
  impact INTEGER NOT NULL,
  sentiment REAL NOT NULL,
  source_type TEXT NOT NULL,
  source_name TEXT NOT NULL,
  is_synthetic INTEGER NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sig_ticker_ts ON signals(ticker, ts_published);
CREATE INDEX IF NOT EXISTS ix_sig_ts ON signals(ts_published);
CREATE TABLE IF NOT EXISTS stress_runs (
  run_id TEXT PRIMARY KEY, ts TEXT NOT NULL, trigger_signal TEXT, payload TEXT NOT NULL
);
"""


class SignalStore:
    def __init__(self, sqlite_path: str | Path | None = None, jsonl_path: str | Path | None = None):
        st = get_config().section("storage")
        self.sqlite_path = resolve(sqlite_path or st.get("sqlite_path", "data/runtime/trident.db"))
        self.jsonl_path = resolve(jsonl_path or st.get("jsonl_path", "data/runtime/signals.jsonl"))
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def reset(self) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM signals")
            self.conn.execute("DELETE FROM stress_runs")
            self.conn.commit()
        self.jsonl_path.write_text("")

    def add(self, signals: list[Signal]) -> None:
        if not signals:
            return
        rows = [(s.signal_id, s.ts_published, s.ts_emitted, s.entity.ticker, s.event_classification.label, s.impact_score,
                 s.sentiment_score, s.source.type, s.source.name, int(s.lineage.is_synthetic), s.model_dump_json()) for s in signals]
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                for s in signals:
                    fh.write(s.model_dump_json() + "\n")

    def query(self, ticker: str | None = None, since: str | None = None, until: str | None = None, event: str | None = None,
              min_impact: int | None = None, include_synthetic: bool = True, limit: int = 200) -> list[dict[str, Any]]:
        sql, args = "SELECT payload FROM signals WHERE 1=1", []
        if ticker:
            sql += " AND ticker = ?"
            args.append(ticker)
        if since:
            sql += " AND ts_published >= ?"
            args.append(since)
        if until:
            sql += " AND ts_published <= ?"
            args.append(until)
        if event:
            sql += " AND event_label = ?"
            args.append(event)
        if min_impact:
            sql += " AND impact >= ?"
            args.append(min_impact)
        if not include_synthetic:
            sql += " AND is_synthetic = 0"
        sql += " ORDER BY ts_published DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [json.loads(r[0]) for r in self.conn.execute(sql, args).fetchall()]

    def get(self, signal_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT payload FROM signals WHERE signal_id = ?", (signal_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self) -> int:
        with self.lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0])

    def add_stress_run(self, run_id: str, ts: str, trigger_signal: str | None, payload: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO stress_runs VALUES (?,?,?,?)", (run_id, ts, trigger_signal, json.dumps(payload, default=str)))
            self.conn.commit()

    def stress_runs(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT payload FROM stress_runs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

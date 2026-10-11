import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TRIDENT_STORAGE_SQLITE_PATH", str(ROOT / "data" / "runtime" / "test.db"))
os.environ.setdefault("TRIDENT_STORAGE_JSONL_PATH", str(ROOT / "data" / "runtime" / "test_signals.jsonl"))
# tests always use the deterministic offline backends
os.environ.setdefault("TRIDENT_ENGINE_SENTIMENT_BACKEND", "lexicon")
os.environ.setdefault("TRIDENT_ENGINE_EVENT_BACKEND", "rules")

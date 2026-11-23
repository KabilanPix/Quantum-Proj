import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import db

def make_snapshot():
    """Copies current active SQLite factory database to demo_snapshot.db for standalone cloud deployment."""
    db.init_db()
    if config.DB_PATH.exists():
        shutil.copy2(config.DB_PATH, config.DEMO_SNAPSHOT_DB)
        print(f"[Snapshot] Successfully backed up {config.DB_PATH} -> {config.DEMO_SNAPSHOT_DB}")
    else:
        print(f"[Snapshot Warning] {config.DB_PATH} not found.")

if __name__ == "__main__":
    make_snapshot()

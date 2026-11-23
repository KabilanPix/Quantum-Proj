from pathlib import Path
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "factory.db"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
MODEL_PATH = BASE_DIR / "ml" / "model.joblib"
REAL_DATA_PATH = EXTERNAL_DATA_DIR / "ai4i2020.csv"
MODEL_REAL_PATH = BASE_DIR / "ml" / "model_real.joblib"
REAL_METRICS_PATH = DATA_DIR / "real_metrics.json"
MODEL_COMPARISON_PATH = DATA_DIR / "model_comparison.csv"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.csv"
BENCHMARK_SUMMARY_PATH = DATA_DIR / "benchmark_summary.csv"
DEMO_SNAPSHOT_DB = DATA_DIR / "demo_snapshot.db"

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 5000
API_URL = f"http://{API_HOST}:{API_PORT}"

# Factory & Simulator Configuration
MACHINES = ["M1", "M2", "M3"]
SIM_INTERVAL_SEC = 1.0

# Pipeline Configuration
PIPELINE_INTERVAL_SEC = 5.0

# Normal Operating Ranges
TEMP_NORMAL = (55.0, 65.0)
VIB_NORMAL = (0.2, 0.5)
POWER_NORMAL = (3.0, 4.0)

# Status Thresholds
TEMP_CRITICAL = 85.0
VIB_CRITICAL = 1.5
TEMP_WARNING = 75.0
VIB_WARNING = 1.0
POWER_IDLE = 0.3

# ML Configuration
WINDOW_SIZE = 30
RISK_HIGH = 0.7
RISK_MED = 0.4

# Alert Configuration
ALERT_COOLDOWN_SEC = 300  # 5 minutes cooldown per machine
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def determine_status(temperature: float, vibration: float, power: float) -> str:
    """Computes machine status based on standard config thresholds."""
    if power < POWER_IDLE:
        return "IDLE"
    if temperature > TEMP_CRITICAL or vibration > VIB_CRITICAL:
        return "CRITICAL"
    if temperature > TEMP_WARNING or vibration > VIB_WARNING:
        return "WARNING"
    return "HEALTHY"

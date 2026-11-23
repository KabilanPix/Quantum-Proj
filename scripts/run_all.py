import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import signal
from config import API_URL, API_HOST, API_PORT

def run_all():
    print("=" * 60)
    print("   QUANTUM DIGITAL TWIN - SMART FACTORY SYSTEM LAUNCHER   ")
    print("=" * 60)
    
    python_exe = sys.executable
    processes = []

    try:
        # 1. Start Flask API
        print("[1/4] Starting REST API server...")
        api_proc = subprocess.Popen(
            [python_exe, "api.py"],
            cwd=str(PROJECT_ROOT)
        )
        processes.append(api_proc)

        # 2. Wait for API /health endpoint
        print(f"Waiting for API to become ready at {API_URL}/health ...")
        api_ready = False
        for _ in range(30):
            try:
                r = requests.get(f"{API_URL}/health", timeout=1.0)
                if r.status_code == 200:
                    api_ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not api_ready:
            print("ERROR: API failed to respond within 15 seconds. Terminating.")
            sys.exit(1)
        print("API is UP and Healthy!")

        # 3. Start IoT Sensor Simulator (degrade in 120s for fast live demo)
        print("[2/4] Launching realistic Sensor Simulator (scenario=mixed, noise=medium)...")
        sim_proc = subprocess.Popen(
            [python_exe, "simulator.py", "--reset-db", "--degrade-seconds", "120", "--scenario", "mixed"],
            cwd=str(PROJECT_ROOT)
        )
        processes.append(sim_proc)

        # 4. Start Automated ML-to-Quantum Pipeline Worker
        print("[3/4] Launching Automated ML-to-Quantum Pipeline Worker...")
        pipe_proc = subprocess.Popen(
            [python_exe, "pipeline.py"],
            cwd=str(PROJECT_ROOT)
        )
        processes.append(pipe_proc)

        # 5. Start Streamlit Dashboard
        print("[4/4] Launching Streamlit Digital Twin Dashboard...")
        dash_proc = subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.headless", "true"],
            cwd=str(PROJECT_ROOT)
        )
        processes.append(dash_proc)

        print("\n" + "=" * 60)
        print("   ALL SERVICES RUNNING SUCCESSFULLY!")
        print(f"   REST API:   http://{API_HOST}:{API_PORT}")
        print("   Dashboard:  http://localhost:8501")
        print("=" * 60)
        print("Press Ctrl+C to terminate all services.\n")

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Shutdown] Received interrupt signal. Terminating child processes...")
    finally:
        for p in processes:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                else:
                    p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in processes:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        print("[Shutdown] All processes cleanly closed. Bye!")

if __name__ == "__main__":
    run_all()

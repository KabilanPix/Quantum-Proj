import argparse
import random
import time
import requests
import db
from config import API_URL, SIM_INTERVAL_SEC

def run_simulator(
    degrade_seconds: float = 240.0,
    interval: float = SIM_INTERVAL_SEC,
    reset_db: bool = False,
    scenario: str = "mixed",
    noise_level: str = "medium"
):
    """
    Upgraded realistic factory simulator:
    Scenarios:
      - 'normal': all machines stay healthy.
      - 'bearing_wear': vibration grows exponentially, temperature rises slowly.
      - 'overheating': temperature climbs quickly (cooling failure), vibration stays normal.
      - 'power_surge': random power spikes, then temperature rises.
      - 'mixed': M1 healthy, M2 idle/run cycles with power surge, M3 bearing wear (default demo).
    Realism (noise_level: low | medium | high):
      - Gaussian noise
      - Sensor dropouts (random skipped posts)
      - Outlier spikes
      - Sensor drift
    Maintenance events:
      - When failure reached, machine enters REPAIR (down 20s), then resets.
      - Logs to `events` table (and via API if accessible, or direct db fallback).
    """
    if reset_db:
        print("[Simulator] Clearing existing database readings...")
        try:
            db.clear_readings()
        except Exception as e:
            print(f"[Simulator] Warning: Failed to clear database: {e}")

    print(f"[Simulator] Starting simulation: scenario='{scenario}', noise='{noise_level}', interval={interval}s, degrade_seconds={degrade_seconds}s")

    # Noise multipliers
    noise_mult = {"low": 0.5, "medium": 1.0, "high": 2.2}.get(noise_level, 1.0)
    dropout_prob = {"low": 0.0, "medium": 0.02, "high": 0.08}.get(noise_level, 0.02)
    spike_prob = {"low": 0.01, "medium": 0.04, "high": 0.10}.get(noise_level, 0.04)

    # Machine states
    # state: "HEALTHY", "DEGRADING", "CRITICAL_HOLD", "REPAIR"
    machines_state = {
        "M1": {"state": "HEALTHY", "elapsed": 0.0, "timer": 0.0, "drift_temp": 0.0, "drift_vib": 0.0},
        "M2": {"state": "HEALTHY", "elapsed": 0.0, "timer": 0.0, "drift_temp": 0.0, "drift_vib": 0.0},
        "M3": {"state": "DEGRADING", "elapsed": 0.0, "timer": 0.0, "drift_temp": 0.0, "drift_vib": 0.0}
    }

    # Assign initial states according to scenario
    if scenario == "normal":
        for m in machines_state:
            machines_state[m]["state"] = "HEALTHY"
    elif scenario in ["bearing_wear", "overheating", "power_surge"]:
        machines_state["M1"]["state"] = "HEALTHY"
        machines_state["M2"]["state"] = "HEALTHY"
        machines_state["M3"]["state"] = "DEGRADING"

    cycle = 0
    while True:
        cycle += 1
        current_time = time.time()
        readings = []

        for m_id in ["M1", "M2", "M3"]:
            st = machines_state[m_id]

            # Sensor dropouts
            if random.random() < dropout_prob:
                # Dropout: skip reading
                continue

            # Slow drift
            st["drift_temp"] += random.uniform(-0.02, 0.02) * noise_mult
            st["drift_temp"] = max(-3.0, min(3.0, st["drift_temp"]))
            st["drift_vib"] += random.uniform(-0.001, 0.001) * noise_mult
            st["drift_vib"] = max(-0.1, min(0.1, st["drift_vib"]))

            # Handle state machine & scenarios
            curr_state = st["state"]

            # Failure mode definition for M3 or active degrading machines
            active_mode = scenario
            if scenario == "mixed":
                active_mode = "bearing_wear" if m_id == "M3" else "normal"

            if curr_state == "REPAIR":
                st["timer"] += interval
                # Machine down during repair
                temp = 35.0 + random.gauss(0, 0.5) * noise_mult
                vib = 0.02 + abs(random.gauss(0, 0.005))
                power = 0.05 + abs(random.gauss(0, 0.01))
                if st["timer"] >= 20.0:
                    # Repaired!
                    st["state"] = "HEALTHY" if scenario == "normal" else "DEGRADING"
                    st["elapsed"] = 0.0
                    st["timer"] = 0.0
                    msg = f"Machine {m_id} maintenance completed. Returned to service."
                    print(f"\n[Simulator Event] >>> {msg} <<<\n")
                    try:
                        db.insert_event(m_id, current_time, "MAINTENANCE_COMPLETED", msg)
                    except Exception:
                        pass
                readings.append({"machine_id": m_id, "temperature": round(temp, 2), "vibration": round(vib, 3), "power": round(power, 2), "timestamp": current_time, "source": "simulator"})
                continue

            if curr_state == "CRITICAL_HOLD":
                st["timer"] += interval
                temp = 96.0 + random.gauss(0, 1.2) * noise_mult
                vib = 2.1 + random.gauss(0, 0.08) * noise_mult
                power = 4.4 + random.gauss(0, 0.15) * noise_mult
                if st["timer"] >= 10.0:
                    st["state"] = "REPAIR"
                    st["timer"] = 0.0
                    msg = f"Machine {m_id} reached breakdown limit. Shutting down for 20s repair."
                    print(f"\n[Simulator Event] >>> {msg} <<<\n")
                    try:
                        db.insert_event(m_id, current_time, "BREAKDOWN_SHUTDOWN", msg)
                    except Exception:
                        pass
                readings.append({"machine_id": m_id, "temperature": round(temp, 2), "vibration": round(vib, 3), "power": round(power, 2), "timestamp": current_time, "source": "simulator"})
                continue

            # M2 Idle cycling in mixed or normal
            if m_id == "M2":
                cycle_time = current_time % 105.0
                if cycle_time > 90.0:
                    temp = 48.0 + random.gauss(0, 0.8) * noise_mult + st["drift_temp"]
                    vib = 0.1 + random.gauss(0, 0.02) * noise_mult + st["drift_vib"]
                    power = max(0.05, 0.15 + random.gauss(0, 0.03))
                else:
                    surge = 2.5 if (scenario == "mixed" and random.random() < 0.05) else 0.0
                    temp = 61.0 + random.gauss(0, 1.0) * noise_mult + (0.5 * surge) + st["drift_temp"]
                    vib = 0.38 + random.gauss(0, 0.03) * noise_mult + st["drift_vib"]
                    power = 3.6 + random.gauss(0, 0.1) * noise_mult + surge
                readings.append({"machine_id": m_id, "temperature": round(temp, 2), "vibration": round(max(0.01, vib), 3), "power": round(max(0.05, power), 2), "timestamp": current_time, "source": "simulator"})
                continue

            # M1 Healthy
            if m_id == "M1" or curr_state == "HEALTHY":
                temp = 60.0 + random.gauss(0, 1.0) * noise_mult + st["drift_temp"]
                vib = 0.35 + random.gauss(0, 0.03) * noise_mult + st["drift_vib"]
                power = 3.5 + random.gauss(0, 0.08) * noise_mult
                readings.append({"machine_id": m_id, "temperature": round(temp, 2), "vibration": round(max(0.01, vib), 3), "power": round(max(0.05, power), 2), "timestamp": current_time, "source": "simulator"})
                continue

            # Degrading Machine (e.g. M3)
            st["elapsed"] += interval
            prog = min(1.0, st["elapsed"] / max(1.0, degrade_seconds))

            # Spikes
            temp_spike = random.uniform(2.0, 5.0) if random.random() < spike_prob else 0.0
            vib_spike = random.uniform(0.2, 0.5) if random.random() < spike_prob else 0.0
            pow_spike = random.uniform(0.8, 2.0) if random.random() < spike_prob else 0.0

            if active_mode == "bearing_wear":
                # Exponential vibration, slow temperature rise
                vib = 0.35 + 1.8 * (prog ** 1.8) + random.gauss(0, 0.04) * noise_mult + vib_spike
                temp = 60.0 + 26.0 * (prog ** 1.1) + random.gauss(0, 1.0) * noise_mult + temp_spike
                power = 3.5 + 0.9 * prog + random.gauss(0, 0.1) * noise_mult
            elif active_mode == "overheating":
                # Temperature climbs quickly, vibration stays normal
                temp = 60.0 + 38.0 * (prog ** 0.8) + random.gauss(0, 1.0) * noise_mult + temp_spike
                vib = 0.35 + random.gauss(0, 0.04) * noise_mult + (0.1 * prog)
                power = 3.5 + 0.6 * prog + random.gauss(0, 0.1) * noise_mult
            elif active_mode == "power_surge":
                # Frequent power surges, followed by thermal buildup
                pow_surge = random.uniform(1.5, 3.5) if random.random() < 0.25 else 0.0
                power = 3.5 + 1.2 * prog + pow_surge + random.gauss(0, 0.15) * noise_mult
                temp = 60.0 + 32.0 * (prog ** 1.2) + (0.8 * pow_surge) + random.gauss(0, 1.0) * noise_mult
                vib = 0.35 + 0.5 * prog + random.gauss(0, 0.05) * noise_mult
            else:
                # Default mixed / linear
                temp = 60.0 + 35.0 * prog + random.gauss(0, 1.0) * noise_mult + temp_spike
                vib = 0.35 + 1.75 * (prog ** 1.3) + random.gauss(0, 0.04) * noise_mult + vib_spike
                power = 3.5 + 0.9 * prog + random.gauss(0, 0.1) * noise_mult

            temp += st["drift_temp"]
            vib += st["drift_vib"]

            # Check if critical threshold reached
            if temp >= 85.0 or vib >= 1.5:
                st["state"] = "CRITICAL_HOLD"
                st["timer"] = 0.0
                msg = f"Machine {m_id} failure detected ({temp:.1f}C, {vib:.2f}g) under {active_mode}."
                print(f"\n[Simulator Event] >>> {msg} <<<\n")
                try:
                    db.insert_event(m_id, current_time, "FAILURE_DETECTED", msg)
                except Exception:
                    pass

            readings.append({
                "machine_id": m_id,
                "temperature": round(temp, 2),
                "vibration": round(max(0.01, vib), 3),
                "power": round(max(0.05, power), 2),
                "timestamp": current_time,
                "source": "simulator"
            })

        # Send batch to API
        sent_statuses = []
        for r in readings:
            try:
                resp = requests.post(f"{API_URL}/sensor-data", json=r, timeout=2.0)
                if resp.status_code == 201:
                    status = resp.json().get("status", "?")
                    sent_statuses.append(f"{r['machine_id']}:{status}({r['temperature']}C,{r['vibration']}g)")
                else:
                    sent_statuses.append(f"{r['machine_id']}:ERR_{resp.status_code}")
            except Exception:
                # Direct DB fallback if API is not running
                try:
                    from config import determine_status
                    st_val = determine_status(r["temperature"], r["vibration"], r["power"])
                    db.insert_reading(r["machine_id"], r["timestamp"], r["temperature"], r["vibration"], r["power"], st_val, source="simulator")
                    sent_statuses.append(f"{r['machine_id']}:DB_{st_val}")
                except Exception:
                    sent_statuses.append(f"{r['machine_id']}:OFFLINE")

        if cycle % 5 == 0 or any("CRITICAL" in s or "WARNING" in s for s in sent_statuses):
            print(f"[Simulator] Cycle {cycle:04d} | " + " | ".join(sent_statuses))

        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistic Smart Factory Digital Twin Sensor Simulator")
    parser.add_argument("--degrade-seconds", type=float, default=240.0, help="Seconds for machine degradation cycle")
    parser.add_argument("--interval", type=float, default=SIM_INTERVAL_SEC, help="Simulation loop interval in seconds")
    parser.add_argument("--reset-db", action="store_true", help="Clear readings table before starting")
    parser.add_argument("--scenario", type=str, default="mixed", choices=["normal", "bearing_wear", "overheating", "power_surge", "mixed"], help="Failure scenario mode")
    parser.add_argument("--noise-level", type=str, default="medium", choices=["low", "medium", "high"], help="Environmental sensor noise intensity")
    args = parser.parse_args()

    run_simulator(
        degrade_seconds=args.degrade_seconds,
        interval=args.interval,
        reset_db=args.reset_db,
        scenario=args.scenario,
        noise_level=args.noise_level
    )

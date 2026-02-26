"""
VOC Biometric Pi Client  —  Production Ready
=============================================

Runs on Raspberry Pi. Reads MQ6 + MEMS sensors via ADS1115 ADC,
computes statistical features, calls the remote FastAPI API for
person identification, and displays the result.

Hardware wiring (ADS1115):
    MQ6  analog out  → ADS1115 A0
    MEMS analog out  → ADS1115 A1
    ADS1115 VDD      → Pi 3.3V  (pin 1)
    ADS1115 GND      → Pi GND   (pin 6)
    ADS1115 SDA      → Pi GPIO2 (pin 3)
    ADS1115 SCL      → Pi GPIO3 (pin 5)

Hand sensor (IR proximity):
    OUT → GPIO 17    VCC → 3.3V    GND → GND

Fan (via MOSFET relay):
    Gate → GPIO 27   Fan power → external supply

Usage:
    # Simulate (no hardware)
    python client.py --server http://YOUR_IP:8000 --simulate

    # Real hardware
    python client.py --server http://YOUR_IP:8000 --device pi-lab-01

    # Adjust sampling
    python client.py --server http://YOUR_IP:8000 --window 20 --rate 0.05
"""

import argparse
import time
import json
import logging
import sys
import signal
import requests
import numpy as np
from datetime import datetime
from scipy import stats as sp_stats

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voc-client")


# ── CLI Arguments ────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="VOC Biometric Pi Client — identify persons from sensor data"
    )
    p.add_argument("--server",    default="http://localhost:8000",
                   help="API server URL  (default: http://localhost:8000)")
    p.add_argument("--device",    default="pi-001",
                   help="Device identifier sent to the server")
    p.add_argument("--window",    type=int,   default=30,
                   help="Sampling window in seconds  (default: 30)")
    p.add_argument("--rate",      type=float, default=0.1,
                   help="Seconds between samples  (default: 0.1)")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Min confidence for 'identified' display  (default: 0.50)")
    p.add_argument("--rounds",    type=int,   default=0,
                   help="Number of rounds to run (0 = infinite)  (default: 0)")
    p.add_argument("--flush-sec", type=int,   default=15,
                   help="Fan flush duration between rounds  (default: 15)")
    p.add_argument("--hand-pin",  type=int,   default=17,
                   help="GPIO pin for hand-detection IR sensor  (default: 17)")
    p.add_argument("--fan-pin",   type=int,   default=27,
                   help="GPIO pin for fan MOSFET/relay  (default: 27)")
    p.add_argument("--simulate",  action="store_true",
                   help="Run with simulated sensors (no hardware needed)")
    p.add_argument("--no-hand",   action="store_true",
                   help="Skip hand detection (auto-start each round)")
    p.add_argument("--no-fan",    action="store_true",
                   help="Skip fan flush between rounds")
    p.add_argument("--log-file",  default="predictions_log.jsonl",
                   help="Path to write prediction log  (default: predictions_log.jsonl)")
    return p.parse_args()


# ── Statistics (must match server's engineer_features) ───────────
def compute_stats(samples: list[float]) -> dict:
    """
    Compute statistical features from a list of voltage readings.
    These MUST match what the server's SensorReading schema expects.
    """
    arr = np.array(samples, dtype=np.float64)

    q75, q25 = np.percentile(arr, [75, 25])
    skewness = float(sp_stats.skew(arr))  if len(arr) > 2 else 0.0
    kurtosis = float(sp_stats.kurtosis(arr)) if len(arr) > 2 else 0.0
    cv       = float(arr.std() / (arr.mean() + 1e-9))
    energy   = float(np.sum(arr ** 2))

    return {
        "min":      float(arr.min()),
        "mean":     float(arr.mean()),
        "max":      float(arr.max()),
        "std":      float(arr.std()),
        "median":   float(np.median(arr)),
        "iqr":      float(q75 - q25),
        "skew":     skewness,
        "kurtosis": kurtosis,
        "cv":       cv,
        "energy":   energy,
    }


def build_payload(
    mq6_samples: list[float],
    mems_samples: list[float],
    round_no: int,
    device_id: str,
) -> dict:
    """Build the JSON payload matching the server's SensorReading schema."""
    mq6  = compute_stats(mq6_samples)
    mems = compute_stats(mems_samples)

    return {
        # MQ6 sensor
        "mq6_1_min":    mq6["min"],
        "mq6_1_mean":   mq6["mean"],
        "mq6_1_max":    mq6["max"],
        "mq6_1_std":    mq6["std"],
        "mq6_1_median": mq6["median"],
        "mq6_1_iqr":    mq6["iqr"],
        "mq6_1_skew":   mq6["skew"],
        # MEMS odor sensor
        "mems_odor_1_min":      mems["min"],
        "mems_odor_1_mean":     mems["mean"],
        "mems_odor_1_max":      mems["max"],
        "mems_odor_1_std":      mems["std"],
        "mems_odor_1_median":   mems["median"],
        "mems_odor_1_iqr":      mems["iqr"],
        "mems_odor_1_skew":     mems["skew"],
        "mems_odor_1_kurtosis": mems["kurtosis"],
        "mems_odor_1_cv":       mems["cv"],
        "mems_odor_1_energy":   mems["energy"],
        # Context
        "round_no":  round_no,
        "device_id": device_id,
        "timestamp": datetime.now().isoformat(),
    }


# ── API Calls ────────────────────────────────────────────────────
def health_check(server: str) -> bool:
    """Verify the server is reachable. Returns True on success."""
    try:
        resp = requests.get(f"{server}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        log.info(f"✅ Server OK  →  model={data.get('model', '?')}  ")
        return True
    except requests.exceptions.ConnectionError:
        log.error(f"❌ Cannot reach {server}. Is the server running?")
    except Exception as e:
        log.error(f"❌ Health check failed: {e}")
    return False


def fetch_persons(server: str) -> list[str]:
    """Fetch list of known persons from the server."""
    try:
        resp = requests.get(f"{server}/persons", timeout=5)
        return resp.json().get("persons", [])
    except Exception:
        return []


def call_predict(payload: dict, server: str) -> dict | None:
    """POST sensor payload to /predict. Returns result dict or None."""
    try:
        resp = requests.post(
            f"{server}/predict",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        log.error(f"❌ Cannot reach server at {server}")
    except requests.exceptions.Timeout:
        log.error("❌ Request timed out")
    except requests.exceptions.HTTPError as e:
        log.error(f"❌ Server error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        log.error(f"❌ Unexpected error: {e}")
    return None


# ── Display ──────────────────────────────────────────────────────
def display_result(result: dict, threshold: float):
    """Pretty-print the prediction result to the terminal."""
    person     = result["person"]
    confidence = result["confidence"]
    status     = result["status"]
    latency    = result["latency_ms"]

    bar_len = int(confidence * 30)
    bar     = "█" * bar_len + "░" * (30 - bar_len)

    if status == "identified" and confidence >= threshold:
        icon = "✅"
    elif status == "uncertain":
        icon = "⚠️ "
    else:
        icon = "❓"

    print(f"\n{'─' * 56}")
    print(f"  {icon}  Person     : {person}")
    print(f"      Confidence : [{bar}] {confidence:.1%}")
    print(f"      Status     : {status.upper()}")
    print(f"      Latency    : {latency:.1f} ms")
    if confidence < threshold:
        print(f"      ⚠️  Below threshold ({threshold:.0%}) — unreliable")
    print(f"{'─' * 56}")

    # Top 3 candidates
    top3 = sorted(result.get("all_probs", {}).items(),
                  key=lambda x: x[1], reverse=True)[:3]
    if top3:
        print("  Top candidates:")
        for name, prob in top3:
            b = int(prob * 20)
            print(f"    {name:<22} {'█' * b}{'░' * (20 - b)} {prob:.1%}")
    print()


# ── Logging ──────────────────────────────────────────────────────
def log_prediction(path: str, round_no: int, payload: dict, result: dict):
    """Append prediction to a JSONL log file."""
    try:
        with open(path, "a") as f:
            f.write(json.dumps({
                "round_no": round_no,
                "payload":  payload,
                "result":   result,
            }) + "\n")
    except Exception as e:
        log.warning(f"Could not write to log: {e}")


# ── Main Loop ────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── banner ───────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   🔬  VOC BIOMETRIC IDENTIFICATION CLIENT   ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    log.info(f"Server    : {args.server}")
    log.info(f"Device    : {args.device}")
    log.info(f"Window    : {args.window}s  |  Rate: {args.rate}s/sample")
    log.info(f"Threshold : {args.threshold:.0%}")
    log.info(f"Simulate  : {args.simulate}")
    log.info(f"Hand det. : {'OFF' if args.no_hand else f'GPIO {args.hand_pin}'}")
    log.info(f"Fan flush : {'OFF' if args.no_fan  else f'GPIO {args.fan_pin}, {args.flush_sec}s'}")
    print()

    # ── health check ─────────────────────────────────────────────
    if not health_check(args.server):
        log.error("Start the server first:")
        log.error("  cd server/ && uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Show known persons
    persons = fetch_persons(args.server)
    if persons:
        log.info(f"Known persons ({len(persons)}): {', '.join(persons[:6])}{'…' if len(persons) > 6 else ''}")

    # ── init hardware ────────────────────────────────────────────
    from hardware.sensor_reader import SensorReader
    from hardware.hand_controller import HandController
    from hardware.fan_manager import FanManager

    sensor = SensorReader(simulate=args.simulate)
    hand   = HandController(pin=args.hand_pin, simulate=(args.simulate or args.no_hand))
    fan    = FanManager(pin=args.fan_pin, simulate=(args.simulate or args.no_fan))

    if sensor.is_simulated:
        log.info("📡 Running with SIMULATED sensors")
    else:
        log.info("📡 Running with REAL hardware sensors")

    # ── graceful shutdown ────────────────────────────────────────
    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── sampling loop ────────────────────────────────────────────
    round_no = 0
    print("\n🔬 Starting VOC sampling loop. Press Ctrl+C to stop.\n")

    try:
        while not shutdown:
            round_no += 1
            if args.rounds > 0 and round_no > args.rounds:
                log.info(f"Completed {args.rounds} rounds. Exiting.")
                break

            # ── 1. Wait for hand ─────────────────────────────────
            if not args.no_hand:
                print(f"[Round {round_no}] Place hand over the sensor…")
                hand.wait_for_hand()

            # ── 2. Collect samples ───────────────────────────────
            print(f"[Round {round_no}] Sampling for {args.window}s", end="", flush=True)

            def progress(n):
                if n % 10 == 0:
                    print(".", end="", flush=True)

            hand_fn = hand.hand_present if not args.no_hand else None

            mq6_samples, mems_samples = sensor.collect_window(
                duration_sec=args.window,
                rate_sec=args.rate,
                hand_check_fn=hand_fn,
                on_sample_fn=progress,
            )

            n_samples = len(mq6_samples)
            print(f"  {n_samples} samples")

            if n_samples < 5:
                log.warning("Too few samples — skipping this round.")
                continue

            # ── 3. Build payload + call API ──────────────────────
            payload = build_payload(mq6_samples, mems_samples, round_no, args.device)

            log.info(f"Sending round {round_no} → {args.server}/predict …")

            result = call_predict(payload, args.server)

            if result:
                display_result(result, args.threshold)
                log_prediction(args.log_file, round_no, payload, result)
            else:
                log.error("No result from API this round.")

            # ── 4. Fan flush ─────────────────────────────────────
            if not args.no_fan and args.flush_sec > 0:
                log.info(f"🌀 Flushing chamber ({args.flush_sec}s)…")
                fan.flush(args.flush_sec)

    except KeyboardInterrupt:
        pass
    finally:
        # ── cleanup ──────────────────────────────────────────────
        fan.cleanup()
        hand.cleanup()
        print(f"\n⏹  Stopped after {round_no} round{'s' if round_no != 1 else ''}.")
        log.info(f"Predictions logged to {args.log_file}")


if __name__ == "__main__":
    main()

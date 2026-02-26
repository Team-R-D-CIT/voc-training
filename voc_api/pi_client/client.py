"""
VOC Biometric Pi Client
Runs on Raspberry Pi. Reads MQ6 + MEMS sensors, computes stats, calls the API.

Usage:
    pip install requests numpy scipy RPi.GPIO
    python client.py --server http://YOUR_MACHINE_IP:8000 --device pi-lab-01

Pin setup:
    MQ6  analog out  → ADS1115 channel 0  (Pi has no built-in ADC)
    MEMS analog out  → ADS1115 channel 1
    ADS1115 SDA      → Pi GPIO 2 (SDA)
    ADS1115 SCL      → Pi GPIO 3 (SCL)

    Install ADS1115 lib: pip install adafruit-circuitpython-ads1x15
"""

import argparse
import time
import json
import logging
import requests
import numpy as np
from datetime import datetime
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="VOC Biometric Pi Client")
parser.add_argument("--server",    default="http://localhost:8000", help="API server URL")
parser.add_argument("--device",    default="pi-001",               help="Device identifier")
parser.add_argument("--window",    type=int,   default=30,          help="Sampling window in seconds")
parser.add_argument("--rate",      type=float, default=0.1,         help="Seconds between samples")
parser.add_argument("--threshold", type=float, default=0.6,         help="Min confidence to display result")
parser.add_argument("--simulate",  action="store_true",             help="Use simulated sensor data (no hardware)")
args = parser.parse_args()


# ── Sensor setup ─────────────────────────────────────────────
def init_sensors():
    """Initialize ADS1115 ADC for MQ6 and MEMS sensors."""
    try:
        import board
        import busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)

        mq6_channel  = AnalogIn(ads, ADS.P0)
        mems_channel = AnalogIn(ads, ADS.P1)

        log.info("✅ ADS1115 initialized (MQ6 → CH0, MEMS → CH1)")
        return mq6_channel, mems_channel

    except Exception as e:
        log.warning(f"Hardware init failed: {e}")
        log.warning("Falling back to simulated data.")
        return None, None


def read_sensors(mq6_ch, mems_ch):
    """Read one voltage value from each sensor. Returns (mq6_v, mems_v)."""
    mq6_v  = mq6_ch.voltage  if mq6_ch  else None
    mems_v = mems_ch.voltage if mems_ch else None
    return mq6_v, mems_v


def simulate_sensors(t: float):
    """
    Simulate realistic VOC sensor readings for testing without hardware.
    Adds slight drift + noise to mimic real sensor behavior.
    """
    mq6_base  = 3.5 + 0.05 * np.sin(t / 20)
    mems_base = 17.5 + 0.1  * np.sin(t / 15)
    mq6_v     = mq6_base  + np.random.normal(0, 0.01)
    mems_v    = mems_base + np.random.normal(0, 0.05)
    return mq6_v, mems_v


# ── Stats computation ─────────────────────────────────────────
def compute_stats(samples: list) -> dict:
    """
    Compute the same statistical features used during model training.
    This MUST match Step 4 in the notebook exactly.
    """
    arr = np.array(samples)

    q75, q25  = np.percentile(arr, [75, 25])
    skewness  = float(stats.skew(arr)) if len(arr) > 2 else 0.0
    kurtosis  = float(stats.kurtosis(arr)) if len(arr) > 2 else 0.0
    cv        = float(arr.std() / (arr.mean() + 1e-9))
    energy    = float(np.sum(arr ** 2))

    return {
        "min"     : float(arr.min()),
        "mean"    : float(arr.mean()),
        "max"     : float(arr.max()),
        "std"     : float(arr.std()),
        "median"  : float(np.median(arr)),
        "iqr"     : float(q75 - q25),
        "skew"    : skewness,
        "kurtosis": kurtosis,
        "cv"      : cv,
        "energy"  : energy,
    }


def build_payload(mq6_samples: list, mems_samples: list,
                  round_no: int, device_id: str) -> dict:
    mq6_stats  = compute_stats(mq6_samples)
    mems_stats = compute_stats(mems_samples)

    return {
        # MQ6
        "mq6_1_min"   : mq6_stats["min"],
        "mq6_1_mean"  : mq6_stats["mean"],
        "mq6_1_max"   : mq6_stats["max"],
        "mq6_1_std"   : mq6_stats["std"],
        "mq6_1_median": mq6_stats["median"],
        "mq6_1_iqr"   : mq6_stats["iqr"],
        "mq6_1_skew"  : mq6_stats["skew"],
        # MEMS
        "mems_odor_1_min"     : mems_stats["min"],
        "mems_odor_1_mean"    : mems_stats["mean"],
        "mems_odor_1_max"     : mems_stats["max"],
        "mems_odor_1_std"     : mems_stats["std"],
        "mems_odor_1_median"  : mems_stats["median"],
        "mems_odor_1_iqr"     : mems_stats["iqr"],
        "mems_odor_1_skew"    : mems_stats["skew"],
        "mems_odor_1_kurtosis": mems_stats["kurtosis"],
        "mems_odor_1_cv"      : mems_stats["cv"],
        "mems_odor_1_energy"  : mems_stats["energy"],
        # Context
        "round_no"  : round_no,
        "device_id" : device_id,
        "timestamp" : datetime.now().isoformat(),
    }


# ── API call ──────────────────────────────────────────────────
def call_api(payload: dict, server: str) -> dict | None:
    try:
        resp = requests.post(
            f"{server}/predict",
            json=payload,
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        log.error(f"❌ Cannot reach server at {server}. Is it running?")
    except requests.exceptions.Timeout:
        log.error("❌ Request timed out.")
    except requests.exceptions.HTTPError as e:
        log.error(f"❌ Server error: {e.response.status_code} — {e.response.text}")
    return None


# ── Display result ────────────────────────────────────────────
def display_result(result: dict, threshold: float):
    person     = result["person"]
    confidence = result["confidence"]
    status     = result["status"]
    latency    = result["latency_ms"]

    bar_len  = int(confidence * 30)
    bar      = "█" * bar_len + "░" * (30 - bar_len)

    if status == "identified" and confidence >= threshold:
        icon = "✅"
    elif status == "uncertain":
        icon = "⚠️ "
    else:
        icon = "❓"

    print(f"\n{'─'*50}")
    print(f"  {icon}  Person     : {person}")
    print(f"      Confidence : [{bar}] {confidence:.1%}")
    print(f"      Latency    : {latency:.1f}ms")
    if confidence < threshold:
        print(f"      ⚠️  Below threshold ({threshold:.0%}) — result unreliable")
    print(f"{'─'*50}")

    # Top 3 probabilities
    top3 = sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)[:3]
    print("  Top candidates:")
    for name, prob in top3:
        b = int(prob * 20)
        print(f"    {name:<20} {'█'*b}{'░'*(20-b)} {prob:.1%}")


# ── Main loop ─────────────────────────────────────────────────
def main():
    log.info(f"VOC Biometric Client starting")
    log.info(f"Server   : {args.server}")
    log.info(f"Device   : {args.device}")
    log.info(f"Window   : {args.window}s  |  Rate: {args.rate}s/sample")
    log.info(f"Simulate : {args.simulate}")

    # Health check
    try:
        resp = requests.get(f"{args.server}/health", timeout=5)
        log.info(f"✅ Server reachable: {resp.json()}")
    except Exception:
        log.error(f"❌ Server not reachable at {args.server}")
        log.error("   Start the server first: uvicorn main:app --host 0.0.0.0 --port 8000")
        return

    # Init hardware (or simulate)
    if args.simulate:
        mq6_ch, mems_ch = None, None
        log.info("Running in simulation mode.")
    else:
        mq6_ch, mems_ch = init_sensors()
        if mq6_ch is None:
            log.info("Hardware unavailable, switching to simulation mode.")

    round_no   = 1
    t_start_all = time.time()

    print("\n🔬 Starting VOC sampling. Press Ctrl+C to stop.\n")

    try:
        while True:
            mq6_samples  = []
            mems_samples = []

            print(f"[Round {round_no}] Sampling for {args.window}s...", end="", flush=True)
            t_window = time.time()

            while time.time() - t_window < args.window:
                t = time.time() - t_start_all

                if args.simulate or mq6_ch is None:
                    mq6_v, mems_v = simulate_sensors(t)
                else:
                    mq6_v, mems_v = read_sensors(mq6_ch, mems_ch)

                if mq6_v is not None:
                    mq6_samples.append(mq6_v)
                if mems_v is not None:
                    mems_samples.append(mems_v)

                print(".", end="", flush=True)
                time.sleep(args.rate)

            print(f" {len(mq6_samples)} samples")

            if len(mq6_samples) < 5:
                log.warning("Too few samples this round, skipping.")
                continue

            # Build payload and call API
            payload = build_payload(mq6_samples, mems_samples, round_no, args.device)
            log.info(f"Sending round {round_no} to API...")

            result = call_api(payload, args.server)

            if result:
                display_result(result, args.threshold)
                # Save locally as a log
                with open("predictions_log.jsonl", "a") as f:
                    f.write(json.dumps({
                        "round_no": round_no,
                        "payload" : payload,
                        "result"  : result
                    }) + "\n")
            else:
                log.error("No result from API this round.")

            round_no += 1
            print()

    except KeyboardInterrupt:
        print(f"\n\n⏹  Stopped after {round_no - 1} rounds.")
        log.info("Predictions saved to predictions_log.jsonl")


if __name__ == "__main__":
    main()

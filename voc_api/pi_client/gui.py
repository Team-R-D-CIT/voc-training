"""
VOC Biometric Pi GUI  —  Tkinter
=================================
Three-mode interface for Enroll / Verify / Retrain.
Communicates with the FastAPI server over HTTP.

Usage:
    python gui.py --server http://YOUR_IP:8000
    python gui.py --server http://YOUR_IP:8000 --simulate
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
import logging
import argparse
import sys
import requests
import numpy as np
from datetime import datetime
from scipy import stats as sp_stats

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voc-gui")

# ── CLI Args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="VOC Biometric Pi GUI")
parser.add_argument("--server",   default="http://localhost:8000", help="API server URL")
parser.add_argument("--device",   default="pi-001",               help="Device identifier")
parser.add_argument("--window",   type=int,   default=10,         help="Sampling window per round (sec)")
parser.add_argument("--rate",     type=float, default=0.1,        help="Seconds between samples")
parser.add_argument("--rounds",   type=int,   default=5,          help="Number of rounds for enrollment")
parser.add_argument("--simulate", action="store_true",            help="Simulated sensors")
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════
BG_DARK      = "#0F1318"
BG_CARD      = "#181E25"
BG_INPUT     = "#1E2530"
BG_HOVER     = "#232B36"
ACCENT_BLUE  = "#4FA3F7"
ACCENT_GREEN = "#44D7A8"
ACCENT_RED   = "#F7605C"
ACCENT_AMBER = "#F5B942"
TEXT_PRI     = "#E6ECF4"
TEXT_SEC     = "#8896A8"
BORDER       = "#2C3541"
FONT_TITLE   = ("Segoe UI", 22, "bold")
FONT_HEAD    = ("Segoe UI", 13, "bold")
FONT_BODY    = ("Segoe UI", 11)
FONT_MONO    = ("Consolas", 10)
FONT_SMALL   = ("Segoe UI", 9)
FONT_BTN     = ("Segoe UI", 11, "bold")


# ══════════════════════════════════════════════════════════════
#  SENSOR + STATS  (reused from client.py)
# ══════════════════════════════════════════════════════════════
class SensorReader:
    """Reads MQ6 + MEMS sensors via ADS1115, or simulates."""

    def __init__(self, simulate=False):
        self.simulate = simulate
        self.mq6_ch = None
        self.mems_ch = None
        self._t0 = time.time()
        if not simulate:
            self._init_hw()

    def _init_hw(self):
        try:
            import board, busio
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c)
            self.mq6_ch  = AnalogIn(ads, ADS.P0)
            self.mems_ch = AnalogIn(ads, ADS.P1)
            log.info("✅ ADS1115 initialized")
        except Exception as e:
            log.warning(f"⚠ HW init failed: {e} — simulating")
            self.simulate = True

    def read_one(self):
        if self.simulate:
            t = time.time() - self._t0
            mq6  = 3.5  + 0.05 * np.sin(t / 20) + np.random.normal(0, 0.01)
            mems = 17.5 + 0.10 * np.sin(t / 15) + np.random.normal(0, 0.05)
            return float(mq6), float(mems)
        return float(self.mq6_ch.voltage), float(self.mems_ch.voltage)


def compute_stats(samples):
    arr = np.array(samples, dtype=np.float64)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "min":      float(arr.min()),
        "mean":     float(arr.mean()),
        "max":      float(arr.max()),
        "std":      float(arr.std()),
        "median":   float(np.median(arr)),
        "iqr":      float(q75 - q25),
        "skew":     float(sp_stats.skew(arr))     if len(arr) > 2 else 0.0,
        "kurtosis": float(sp_stats.kurtosis(arr)) if len(arr) > 2 else 0.0,
        "cv":       float(arr.std() / (arr.mean() + 1e-9)),
        "energy":   float(np.sum(arr ** 2)),
    }


def build_payload(mq6_samples, mems_samples, round_no=None, device_id=None):
    mq6  = compute_stats(mq6_samples)
    mems = compute_stats(mems_samples)
    return {
        "mq6_1_min": mq6["min"], "mq6_1_mean": mq6["mean"],
        "mq6_1_max": mq6["max"], "mq6_1_std": mq6["std"],
        "mq6_1_median": mq6["median"], "mq6_1_iqr": mq6["iqr"],
        "mq6_1_skew": mq6["skew"],
        "mems_odor_1_min": mems["min"], "mems_odor_1_mean": mems["mean"],
        "mems_odor_1_max": mems["max"], "mems_odor_1_std": mems["std"],
        "mems_odor_1_median": mems["median"], "mems_odor_1_iqr": mems["iqr"],
        "mems_odor_1_skew": mems["skew"],
        "mems_odor_1_kurtosis": mems["kurtosis"],
        "mems_odor_1_cv": mems["cv"], "mems_odor_1_energy": mems["energy"],
        "round_no": round_no, "device_id": device_id or args.device,
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
#  API HELPERS
# ══════════════════════════════════════════════════════════════
def api_get(path):
    return requests.get(f"{args.server}{path}", timeout=10).json()

def api_post(path, data):
    r = requests.post(f"{args.server}{path}", json=data, timeout=30)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════
#  MAIN GUI
# ══════════════════════════════════════════════════════════════
class VOCApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("VOC Biometric System")
        self.geometry("900x680")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self.sensor = SensorReader(simulate=args.simulate)
        self._persons = []
        self._last_payload = None
        self._last_result  = None

        self._build_topbar()
        self.content = tk.Frame(self, bg=BG_DARK)
        self.content.pack(fill="both", expand=True)

        self._show_home()

    # ── topbar ───────────────────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self, bg=BG_CARD, height=52)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Label(bar, text="◈  VOC BIOMETRIC SYSTEM", font=("Consolas", 14, "bold"),
                 bg=BG_CARD, fg=ACCENT_BLUE).pack(side="left", padx=18, pady=12)
        self._status_lbl = tk.Label(bar, text="● IDLE", font=FONT_SMALL,
                                    bg=BG_CARD, fg=TEXT_SEC)
        self._status_lbl.pack(side="right", padx=18)

        mode_txt = "SIMULATED" if args.simulate else "HARDWARE"
        tk.Label(bar, text=mode_txt, font=FONT_SMALL,
                 bg=BG_CARD, fg=ACCENT_AMBER if args.simulate else ACCENT_GREEN
                 ).pack(side="right", padx=8)

    def _set_status(self, text, color=TEXT_SEC):
        self._status_lbl.config(text=f"●  {text}", fg=color)

    # ── helpers ──────────────────────────────────────────────
    def _clear(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _card(self, parent, **kw):
        return tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER,
                        highlightthickness=1, **kw)

    def _btn(self, parent, text, cmd, color=ACCENT_BLUE, width=22, **kw):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg=BG_DARK, activebackground=color,
                      font=FONT_BTN, relief="flat", cursor="hand2",
                      width=width, height=1, **kw)
        return b

    def _lbl(self, parent, text, font=FONT_BODY, fg=TEXT_PRI, **kw):
        return tk.Label(parent, text=text, font=font, bg=BG_CARD, fg=fg, **kw)

    def _fetch_persons(self):
        try:
            self._persons = api_get("/persons").get("persons", [])
        except Exception:
            self._persons = []

    # ═══════════════════════════════════════════════════════════
    #  HOME SCREEN
    # ═══════════════════════════════════════════════════════════
    def _show_home(self):
        self._clear()
        self._set_status("IDLE")
        self._fetch_persons()

        outer = tk.Frame(self.content, bg=BG_DARK)
        outer.place(relx=0.5, rely=0.45, anchor="center")

        tk.Label(outer, text="BIOMETRIC\nIDENTIFICATION",
                 font=FONT_TITLE, bg=BG_DARK, fg=TEXT_PRI,
                 justify="center").pack(pady=(0, 6))
        tk.Label(outer, text="Volatile Organic Compound Identity Platform",
                 font=FONT_BODY, bg=BG_DARK, fg=TEXT_SEC).pack(pady=(0, 30))

        # Connection status
        try:
            h = api_get("/health")
            info_text = f"✅  Server connected  |  Model: {h.get('model','?')}  |  Persons: {h.get('n_persons','?')}"
            info_fg = ACCENT_GREEN
        except Exception:
            info_text = "❌  Server unreachable — check connection"
            info_fg = ACCENT_RED

        tk.Label(outer, text=info_text, font=FONT_SMALL,
                 bg=BG_DARK, fg=info_fg).pack(pady=(0, 30))

        btns = tk.Frame(outer, bg=BG_DARK)
        btns.pack()

        self._btn(btns, "▶  ENROLL NEW USER",
                  self._show_enroll, ACCENT_BLUE, 28).pack(pady=6, ipady=8)
        self._btn(btns, "▶  VERIFY IDENTITY",
                  self._show_verify, ACCENT_GREEN, 28).pack(pady=6, ipady=8)
        self._btn(btns, "▶  RETRAIN MODEL",
                  self._show_retrain, ACCENT_AMBER, 28).pack(pady=6, ipady=8)
        self._btn(btns, "✕  EXIT",
                  self.destroy, ACCENT_RED, 28).pack(pady=6, ipady=8)

    # ═══════════════════════════════════════════════════════════
    #  ENROLL SCREEN
    # ═══════════════════════════════════════════════════════════
    def _show_enroll(self):
        self._clear()
        self._set_status("ENROLL", ACCENT_BLUE)

        left = tk.Frame(self.content, bg=BG_DARK, width=360)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)

        right = tk.Frame(self.content, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=16)

        # ── left: controls ───────────────────────────────────
        tk.Label(left, text="ENROLL NEW USER", font=FONT_HEAD,
                 bg=BG_DARK, fg=ACCENT_BLUE).pack(anchor="w", pady=(0, 12))

        card = self._card(left)
        card.pack(fill="x", pady=(0, 10))

        self._lbl(card, "User ID *", font=FONT_SMALL, fg=TEXT_SEC).pack(
            anchor="w", padx=14, pady=(12, 2))
        uid_var = tk.StringVar()
        tk.Entry(card, textvariable=uid_var, font=FONT_BODY,
                 bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                 relief="flat", highlightthickness=1,
                 highlightbackground=BORDER).pack(fill="x", padx=14, ipady=6)

        self._lbl(card, "User Name", font=FONT_SMALL, fg=TEXT_SEC).pack(
            anchor="w", padx=14, pady=(8, 2))
        name_var = tk.StringVar()
        tk.Entry(card, textvariable=name_var, font=FONT_BODY,
                 bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                 relief="flat", highlightthickness=1,
                 highlightbackground=BORDER).pack(fill="x", padx=14, ipady=6)

        self._lbl(card, f"Rounds: {args.rounds}  |  Window: {args.window}s",
                  font=FONT_SMALL, fg=TEXT_SEC).pack(
            anchor="w", padx=14, pady=(8, 14))

        # Progress
        prog_card = self._card(left)
        prog_card.pack(fill="x", pady=(0, 10))
        self._lbl(prog_card, "PROGRESS", font=FONT_SMALL, fg=TEXT_SEC).pack(
            anchor="w", padx=14, pady=(12, 4))

        prog_bar = ttk.Progressbar(prog_card, length=300, mode="determinate")
        prog_bar.pack(padx=14, pady=(0, 4))

        prog_lbl = tk.Label(prog_card, text="Waiting to start…",
                            font=FONT_SMALL, bg=BG_CARD, fg=TEXT_SEC)
        prog_lbl.pack(padx=14, pady=(0, 14))

        # Buttons
        start_btn = self._btn(left, "▶  START ENROLLMENT", lambda: None, ACCENT_BLUE, 30)
        start_btn.pack(fill="x", pady=(4, 6), ipady=6)

        self._btn(left, "← Back", self._show_home, "#444C58", 30).pack(
            fill="x", ipady=4)

        # ── right: log ───────────────────────────────────────
        tk.Label(right, text="SYSTEM LOG", font=FONT_SMALL,
                 bg=BG_DARK, fg=TEXT_SEC).pack(anchor="w", pady=(0, 4))
        log_box = scrolledtext.ScrolledText(
            right, bg=BG_CARD, fg=TEXT_PRI, font=FONT_MONO,
            relief="flat", highlightthickness=1, highlightbackground=BORDER,
            insertbackground=TEXT_PRI, wrap="word", state="disabled")
        log_box.pack(fill="both", expand=True)

        def log_msg(msg, tag="info"):
            prefix = {"info": "  ", "ok": "✔ ", "warn": "⚠ ", "err": "✘ "}
            colors = {"info": TEXT_PRI, "ok": ACCENT_GREEN,
                      "warn": ACCENT_AMBER, "err": ACCENT_RED}
            log_box.config(state="normal")
            log_box.insert("end", f"{prefix.get(tag, '  ')}{msg}\n")
            log_box.see("end")
            log_box.config(state="disabled")

        # ── capture thread ───────────────────────────────────
        def run_enrollment():
            uid = uid_var.get().strip()
            if not uid:
                messagebox.showwarning("Missing", "Enter a User ID")
                return

            start_btn.config(state="disabled", text="Sampling…")
            log_msg(f"Starting enrollment for {uid}", "info")
            log_msg(f"Rounds: {args.rounds}  |  Window: {args.window}s  |  Rate: {args.rate}s", "info")

            all_rounds = []
            total_rounds = args.rounds

            for rnd in range(1, total_rounds + 1):
                log_msg(f"Round {rnd}/{total_rounds} — sampling {args.window}s…", "info")
                prog_lbl.config(text=f"Round {rnd}/{total_rounds} — sampling…")
                prog_bar["value"] = ((rnd - 1) / total_rounds) * 100

                mq6_s, mems_s = [], []
                t0 = time.time()
                while time.time() - t0 < args.window:
                    m, e = self.sensor.read_one()
                    mq6_s.append(m)
                    mems_s.append(e)
                    time.sleep(args.rate)

                payload = build_payload(mq6_s, mems_s, rnd, args.device)
                all_rounds.append(payload)
                log_msg(f"Round {rnd}: {len(mq6_s)} samples collected", "ok")

                prog_bar["value"] = (rnd / total_rounds) * 100
                self.update_idletasks()

            # Send to server
            prog_lbl.config(text="Sending to server…")
            log_msg(f"Sending {len(all_rounds)} rounds to /enroll…", "info")

            try:
                result = api_post("/enroll", {
                    "user_id": uid,
                    "rounds":  all_rounds,
                })
                log_msg(f"Enrolled! Rounds {result['round_range'][0]}–{result['round_range'][1]}", "ok")
                prog_lbl.config(text="✅ Enrollment complete!")
                self._set_status("ENROLLED", ACCENT_GREEN)
            except Exception as e:
                log_msg(f"Enrollment failed: {e}", "err")
                prog_lbl.config(text="❌ Enrollment failed")
                self._set_status("ERROR", ACCENT_RED)

            start_btn.config(state="normal", text="▶  START ENROLLMENT")

        start_btn.config(command=lambda: threading.Thread(
            target=run_enrollment, daemon=True).start())

    # ═══════════════════════════════════════════════════════════
    #  VERIFY SCREEN
    # ═══════════════════════════════════════════════════════════
    def _show_verify(self):
        self._clear()
        self._set_status("VERIFY", ACCENT_GREEN)
        self._fetch_persons()

        left = tk.Frame(self.content, bg=BG_DARK, width=360)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)

        right = tk.Frame(self.content, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=16)

        # ── left: controls ───────────────────────────────────
        tk.Label(left, text="IDENTITY VERIFICATION", font=FONT_HEAD,
                 bg=BG_DARK, fg=ACCENT_GREEN).pack(anchor="w", pady=(0, 12))

        # Result card
        result_card = self._card(left)
        result_card.pack(fill="x", pady=(0, 10))

        result_icon = tk.Label(result_card, text="❓", font=("Segoe UI", 36),
                               bg=BG_CARD, fg=TEXT_SEC)
        result_icon.pack(pady=(16, 4))

        result_name = tk.Label(result_card, text="—", font=("Segoe UI", 16, "bold"),
                               bg=BG_CARD, fg=TEXT_PRI)
        result_name.pack()

        result_conf_frame = tk.Frame(result_card, bg=BG_CARD)
        result_conf_frame.pack(fill="x", padx=20, pady=(8, 4))

        conf_bar = ttk.Progressbar(result_conf_frame, length=280, mode="determinate")
        conf_bar.pack()

        conf_lbl = tk.Label(result_card, text="Confidence: —",
                            font=FONT_SMALL, bg=BG_CARD, fg=TEXT_SEC)
        conf_lbl.pack(pady=(2, 4))

        status_lbl = tk.Label(result_card, text="Ready to scan",
                              font=FONT_SMALL, bg=BG_CARD, fg=TEXT_SEC)
        status_lbl.pack(pady=(0, 14))

        # Scan button
        scan_btn = self._btn(left, "▶  START SCAN", lambda: None, ACCENT_GREEN, 30)
        scan_btn.pack(fill="x", pady=(4, 8), ipady=6)

        # Feedback section (hidden initially)
        feedback_card = self._card(left)

        fb_lbl = tk.Label(feedback_card, text="Was this correct?",
                          font=FONT_BODY, bg=BG_CARD, fg=TEXT_PRI)
        fb_lbl.pack(pady=(12, 6))

        fb_btns = tk.Frame(feedback_card, bg=BG_CARD)
        fb_btns.pack(pady=(0, 6))

        correct_btn = tk.Button(fb_btns, text="✅ Correct", font=FONT_BTN,
                                bg=ACCENT_GREEN, fg=BG_DARK, relief="flat",
                                width=12, cursor="hand2")
        correct_btn.pack(side="left", padx=4, ipady=4)

        wrong_btn = tk.Button(fb_btns, text="❌ Wrong", font=FONT_BTN,
                              bg=ACCENT_RED, fg=BG_DARK, relief="flat",
                              width=12, cursor="hand2")
        wrong_btn.pack(side="left", padx=4, ipady=4)

        # Correction dropdown (hidden initially)
        correction_frame = tk.Frame(feedback_card, bg=BG_CARD)

        self._lbl(correction_frame, "Actual person:", font=FONT_SMALL, fg=TEXT_SEC
                  ).pack(anchor="w", padx=14, pady=(6, 2))

        actual_var = tk.StringVar()
        person_combo = ttk.Combobox(correction_frame, textvariable=actual_var,
                                     values=self._persons, font=FONT_BODY, width=26)
        person_combo.pack(padx=14, pady=(0, 6))

        submit_fb_btn = self._btn(correction_frame, "Submit Correction",
                                   lambda: None, ACCENT_AMBER, 26)
        submit_fb_btn.pack(padx=14, pady=(0, 14), ipady=4)

        self._btn(left, "← Back", self._show_home, "#444C58", 30).pack(
            fill="x", ipady=4)

        # ── right: log ───────────────────────────────────────
        tk.Label(right, text="PREDICTION LOG", font=FONT_SMALL,
                 bg=BG_DARK, fg=TEXT_SEC).pack(anchor="w", pady=(0, 4))
        log_box = scrolledtext.ScrolledText(
            right, bg=BG_CARD, fg=TEXT_PRI, font=FONT_MONO,
            relief="flat", highlightthickness=1, highlightbackground=BORDER,
            insertbackground=TEXT_PRI, wrap="word", state="disabled")
        log_box.pack(fill="both", expand=True)

        def log_msg(msg, tag="info"):
            prefix = {"info": "  ", "ok": "✔ ", "warn": "⚠ ", "err": "✘ "}
            log_box.config(state="normal")
            log_box.insert("end", f"{prefix.get(tag, '  ')}{msg}\n")
            log_box.see("end")
            log_box.config(state="disabled")

        # ── scan thread ──────────────────────────────────────
        def run_scan():
            scan_btn.config(state="disabled", text="Scanning…")
            feedback_card.pack_forget()
            correction_frame.pack_forget()

            result_icon.config(text="🔬", fg=ACCENT_BLUE)
            result_name.config(text="Scanning…")
            status_lbl.config(text=f"Collecting {args.window}s of data…")
            conf_bar["value"] = 0
            conf_lbl.config(text="Confidence: —")

            log_msg(f"Sampling {args.window}s at {args.rate}s intervals…", "info")

            mq6_s, mems_s = [], []
            t0 = time.time()
            while time.time() - t0 < args.window:
                m, e = self.sensor.read_one()
                mq6_s.append(m)
                mems_s.append(e)
                elapsed = time.time() - t0
                conf_bar["value"] = (elapsed / args.window) * 100
                self.update_idletasks()
                time.sleep(args.rate)

            log_msg(f"Collected {len(mq6_s)} samples", "ok")

            payload = build_payload(mq6_s, mems_s, device_id=args.device)
            self._last_payload = payload

            status_lbl.config(text="Sending to server…")
            log_msg("Calling /predict…", "info")

            try:
                result = api_post("/predict", payload)
                self._last_result = result

                person = result["person"]
                confidence = result["confidence"]
                status = result["status"]

                # Update display
                if status == "identified":
                    result_icon.config(text="✅", fg=ACCENT_GREEN)
                    self._set_status("IDENTIFIED", ACCENT_GREEN)
                else:
                    result_icon.config(text="⚠️", fg=ACCENT_AMBER)
                    self._set_status("UNCERTAIN", ACCENT_AMBER)

                result_name.config(text=person)
                conf_bar["value"] = confidence * 100
                conf_lbl.config(text=f"Confidence: {confidence:.1%}  |  {result['latency_ms']:.0f}ms")
                status_lbl.config(text=f"Status: {status.upper()}")

                log_msg(f"→ {person}  ({confidence:.1%})  [{status}]", "ok")

                # Show top 3
                top3 = sorted(result.get("all_probs", {}).items(),
                              key=lambda x: x[1], reverse=True)[:3]
                for name, prob in top3:
                    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                    log_msg(f"  {name:<20} {bar} {prob:.1%}", "info")

                # Show feedback buttons
                feedback_card.pack(fill="x", pady=(10, 0))

            except Exception as e:
                result_icon.config(text="❌", fg=ACCENT_RED)
                result_name.config(text="Error")
                status_lbl.config(text=str(e)[:60])
                log_msg(f"Prediction failed: {e}", "err")
                self._set_status("ERROR", ACCENT_RED)

            scan_btn.config(state="normal", text="▶  START SCAN")

        scan_btn.config(command=lambda: threading.Thread(
            target=run_scan, daemon=True).start())

        # ── feedback handlers ────────────────────────────────
        def on_correct():
            log_msg("Marked as correct ✅", "ok")
            feedback_card.pack_forget()
            status_lbl.config(text="Feedback: Correct ✅")

        def on_wrong():
            correction_frame.pack(fill="x", padx=0, pady=(0, 14))
            person_combo["values"] = self._persons
            log_msg("Marked as wrong — select actual person", "warn")

        def submit_correction():
            actual = actual_var.get().strip()
            if not actual:
                messagebox.showwarning("Missing", "Select or type the actual person")
                return

            try:
                api_post("/feedback", {
                    "predicted": self._last_result["person"],
                    "actual":    actual,
                    "features":  self._last_payload,
                })
                log_msg(f"Feedback sent: actual={actual}", "ok")
                correction_frame.pack_forget()
                feedback_card.pack_forget()
                status_lbl.config(text=f"Corrected → {actual}")
            except Exception as e:
                log_msg(f"Feedback failed: {e}", "err")

        correct_btn.config(command=on_correct)
        wrong_btn.config(command=on_wrong)
        submit_fb_btn.config(command=submit_correction)

    # ═══════════════════════════════════════════════════════════
    #  RETRAIN SCREEN
    # ═══════════════════════════════════════════════════════════
    def _show_retrain(self):
        self._clear()
        self._set_status("RETRAIN", ACCENT_AMBER)

        left = tk.Frame(self.content, bg=BG_DARK, width=340)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)

        right = tk.Frame(self.content, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=16)

        # ── left: controls ───────────────────────────────────
        tk.Label(left, text="MODEL RETRAINING", font=FONT_HEAD,
                 bg=BG_DARK, fg=ACCENT_AMBER).pack(anchor="w", pady=(0, 12))

        # Parameters card
        param_card = self._card(left)
        param_card.pack(fill="x", pady=(0, 10))

        self._lbl(param_card, "PARAMETERS", font=FONT_SMALL, fg=TEXT_SEC).pack(
            anchor="w", padx=14, pady=(12, 6))

        p_frame = tk.Frame(param_card, bg=BG_CARD)
        p_frame.pack(fill="x", padx=14, pady=(0, 14))

        tk.Label(p_frame, text="Trees:", font=FONT_SMALL, bg=BG_CARD, fg=TEXT_SEC
                 ).grid(row=0, column=0, sticky="w", pady=2)
        trees_var = tk.IntVar(value=200)
        tk.Spinbox(p_frame, from_=50, to=500, increment=50, textvariable=trees_var,
                   font=FONT_BODY, width=8, bg=BG_INPUT, fg=TEXT_PRI,
                   buttonbackground=BG_HOVER).grid(row=0, column=1, padx=(8, 0), pady=2)

        tk.Label(p_frame, text="Test rounds:", font=FONT_SMALL, bg=BG_CARD, fg=TEXT_SEC
                 ).grid(row=1, column=0, sticky="w", pady=2)
        test_var = tk.IntVar(value=3)
        tk.Spinbox(p_frame, from_=1, to=5, textvariable=test_var,
                   font=FONT_BODY, width=8, bg=BG_INPUT, fg=TEXT_PRI,
                   buttonbackground=BG_HOVER).grid(row=1, column=1, padx=(8, 0), pady=2)

        # Status card
        status_card = self._card(left)
        status_card.pack(fill="x", pady=(0, 10))

        self._lbl(status_card, "STATUS", font=FONT_SMALL, fg=TEXT_SEC).pack(
            anchor="w", padx=14, pady=(12, 4))

        train_status_lbl = tk.Label(status_card, text="Ready",
                                     font=FONT_BODY, bg=BG_CARD, fg=TEXT_PRI)
        train_status_lbl.pack(padx=14, pady=(0, 4))

        train_acc_lbl = tk.Label(status_card, text="Accuracy: —",
                                  font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=TEXT_SEC)
        train_acc_lbl.pack(padx=14, pady=(0, 14))

        # Buttons
        train_btn = self._btn(left, "▶  START RETRAINING", lambda: None, ACCENT_AMBER, 30)
        train_btn.pack(fill="x", pady=(4, 6), ipady=6)

        self._btn(left, "← Back", self._show_home, "#444C58", 30).pack(
            fill="x", ipady=4)

        # ── right: live log ──────────────────────────────────
        tk.Label(right, text="TRAINING LOG", font=FONT_SMALL,
                 bg=BG_DARK, fg=TEXT_SEC).pack(anchor="w", pady=(0, 4))
        log_box = scrolledtext.ScrolledText(
            right, bg=BG_CARD, fg=TEXT_PRI, font=FONT_MONO,
            relief="flat", highlightthickness=1, highlightbackground=BORDER,
            insertbackground=TEXT_PRI, wrap="word", state="disabled")
        log_box.pack(fill="both", expand=True)

        def log_msg(msg):
            log_box.config(state="normal")
            log_box.insert("end", msg + "\n")
            log_box.see("end")
            log_box.config(state="disabled")

        # ── retrain thread ───────────────────────────────────
        def run_retrain():
            train_btn.config(state="disabled", text="Training…")
            train_status_lbl.config(text="Starting…", fg=ACCENT_AMBER)
            train_acc_lbl.config(text="Accuracy: —", fg=TEXT_SEC)
            log_box.config(state="normal")
            log_box.delete("1.0", "end")
            log_box.config(state="disabled")
            self._set_status("TRAINING", ACCENT_AMBER)

            # Kick off retraining
            try:
                result = api_post("/retrain", {
                    "n_estimators": trees_var.get(),
                    "test_rounds":  test_var.get(),
                })
                job_id = result["job_id"]
                log_msg(f"Training started (job {job_id})")
            except Exception as e:
                log_msg(f"❌ Failed to start: {e}")
                train_btn.config(state="normal", text="▶  START RETRAINING")
                train_status_lbl.config(text="Failed", fg=ACCENT_RED)
                self._set_status("ERROR", ACCENT_RED)
                return

            # Poll for updates
            seen_lines = 0
            while True:
                try:
                    status = api_get("/retrain/status")
                except Exception:
                    time.sleep(2)
                    continue

                # Show new log lines
                logs = status.get("logs", [])
                for line in logs[seen_lines:]:
                    log_msg(line)
                seen_lines = len(logs)

                job_status = status.get("status", "unknown")
                train_status_lbl.config(text=job_status.upper())

                if job_status == "done":
                    acc = status.get("accuracy")
                    if acc is not None:
                        train_acc_lbl.config(text=f"Accuracy: {acc:.1%}",
                                              fg=ACCENT_GREEN)
                    train_status_lbl.config(text="✅ COMPLETE", fg=ACCENT_GREEN)
                    self._set_status("RETRAINED", ACCENT_GREEN)
                    log_msg("═" * 40)
                    log_msg("✅ Model retrained and hot-reloaded!")
                    break

                elif job_status == "failed":
                    err = status.get("error", "Unknown error")
                    train_status_lbl.config(text="❌ FAILED", fg=ACCENT_RED)
                    train_acc_lbl.config(text="—", fg=ACCENT_RED)
                    self._set_status("FAILED", ACCENT_RED)
                    log_msg(f"❌ Training failed: {err}")
                    break

                self.update_idletasks()
                time.sleep(2)

            train_btn.config(state="normal", text="▶  START RETRAINING")

        train_btn.config(command=lambda: threading.Thread(
            target=run_retrain, daemon=True).start())


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = VOCApp()
    app.mainloop()

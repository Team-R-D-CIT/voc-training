# VOC Biometric Identification System

Real-time person identification from VOC (Volatile Organic Compound) sensor readings.
Raspberry Pi collects MQ6 + MEMS sensor data → sends to FastAPI server → returns predicted identity.

**Includes:** Enrollment, Verification with feedback, and Model Retraining — all through a GUI or API.

---

## Architecture

```
┌─────────────────────────────┐      HTTP       ┌──────────────────────────────────┐
│     RASPBERRY PI (client)   │ ◄──────────────►│   YOUR MACHINE (server)          │
│                             │                 │                                  │
│  gui.py  (Tkinter GUI)     │   POST /enroll  │  POST /enroll    → append CSV    │
│   ├─ Enroll New User       │   POST /predict │  POST /predict   → inference     │
│   ├─ Verify Identity       │   POST /feedback│  POST /feedback  → correct data  │
│   └─ Retrain Model         │   POST /retrain │  POST /retrain   → background    │
│                             │   GET /status   │  GET  /retrain/status → logs     │
│  client.py (headless CLI)  │                 │                                  │
│                             │                 │  model.pkl, label_encoder.pkl    │
│  hardware/                  │                 │  → hot-reloaded after retrain    │
│   ├─ sensor_reader.py (ADC)│                 │                                  │
│   ├─ hand_controller.py    │                 │  data.csv ← training data        │
│   └─ fan_manager.py        │                 │                                  │
└─────────────────────────────┘                 └──────────────────────────────────┘
```

---

## Project Structure

```
voc_api/
├── server/
│   ├── main.py                  ← FastAPI server (predict + enroll + feedback + retrain)
│   ├── requirements.txt
│   └── model/                   ← Model artifacts (auto-backed up before retrain)
│       ├── model.pkl
│       ├── label_encoder.pkl
│       ├── top_features.pkl
│       ├── metadata.json
│       └── backups/             ← Auto-created on retrain
│
└── pi_client/
    ├── gui.py                   ← Tkinter GUI (Enroll / Verify / Retrain)
    ├── client.py                ← Headless CLI client (continuous mode)
    ├── requirements.txt
    └── hardware/
        ├── sensor_reader.py     ← ADS1115 ADC for MQ6 + MEMS
        ├── hand_controller.py   ← IR proximity hand detection
        └── fan_manager.py       ← Fan flush via MOSFET/relay
```

---

## Quick Start

### 1. Server (Your Machine)

```bash
cd voc_api/server/
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify: http://localhost:8000/docs → Swagger UI with all endpoints

### 2. GUI Client (Raspberry Pi or locally)

```bash
cd voc_api/pi_client/
pip install -r requirements.txt

# On Pi with hardware:
pip install RPi.GPIO adafruit-circuitpython-ads1x15 adafruit-blinka
python gui.py --server http://YOUR_MACHINE_IP:8000

# Locally with simulated sensors:
python gui.py --server http://localhost:8000 --simulate
```

### 3. Headless CLI Client (alternative)

```bash
python client.py --server http://localhost:8000 --simulate --rounds 3
```

---

## GUI Screens

### Enroll New User
- Enter User ID and Name
- Collects N rounds of sensor data (configurable)
- Live progress bar per round
- POSTs all rounds to `/enroll` → appended to `data.csv`

### Verify Identity
- Collects one round of sensor data
- POSTs to `/predict` → displays person + confidence bar + top candidates
- **✅ Correct** / **❌ Wrong** feedback buttons
- If Wrong: select actual person from dropdown → POSTs to `/feedback` → appended to CSV with true label

### Retrain Model
- Configurable: number of trees, test rounds held out
- Kicks off `/retrain` in background thread
- Live scrolling log window shows: data loading, feature engineering, CV folds, final accuracy, per-person breakdown
- Model hot-reloaded in memory — no server restart needed
- Old model auto-backed up

---

## API Endpoints

| Method | Path               | Description                           |
|--------|--------------------|---------------------------------------|
| GET    | `/`                | Server info + model summary           |
| GET    | `/health`          | Health check                          |
| GET    | `/persons`         | List of known persons                 |
| POST   | `/predict`         | Inference from sensor stats           |
| POST   | `/enroll`          | Enroll new user (multi-round)         |
| POST   | `/feedback`        | Correction feedback (wrong prediction)|
| POST   | `/retrain`         | Start background retraining           |
| GET    | `/retrain/status`  | Poll training progress + logs         |
| GET    | `/retrain/logs/stream` | SSE stream of training logs       |
| GET    | `/docs`            | Swagger UI                            |

---

## GUI Flags

| Flag           | Default                | Description                     |
|----------------|------------------------|---------------------------------|
| `--server`     | `http://localhost:8000` | API server URL                  |
| `--device`     | `pi-001`               | Device ID sent to server        |
| `--window`     | `10`                   | Seconds per sampling round      |
| `--rate`       | `0.1`                  | Seconds between samples         |
| `--rounds`     | `5`                    | Rounds per enrollment           |
| `--simulate`   | `False`                | Simulated sensors               |

---

## Data Flow

```
ENROLL:
  Pi GUI → collect N rounds → POST /enroll → append to data.csv

VERIFY:
  Pi GUI → collect 1 round → POST /predict → display result
    → ✅ Correct (done)
    → ❌ Wrong → select actual → POST /feedback → append corrected row to CSV

RETRAIN:
  Pi GUI → POST /retrain → server background thread:
    load data.csv → engineer features → train/test split by round
    → 5-fold CV → final fit → evaluate → save artifacts → hot-reload
    → Pi polls /retrain/status for live logs
```

---

## Hardware Wiring

```
ADS1115 ADC:
  MQ6 analog out   → A0        MEMS analog out → A1
  VDD → 3.3V (pin 1)           GND → pin 6
  SDA → GPIO2 (pin 3)          SCL → GPIO3 (pin 5)

Hand IR sensor:
  OUT → GPIO 17 (pin 11)       VCC → 3.3V    GND → GND

Fan (MOSFET/relay):
  Gate → GPIO 27 (pin 13)      Fan power → external supply
```

# VOC Biometric API

Real-time person identification from VOC sensor readings.
Raspberry Pi collects data → sends to FastAPI server → returns predicted identity.

---

## Project Structure

```
voc_api/
├── server/
│   ├── main.py              ← FastAPI server (runs on YOUR machine)
│   ├── requirements.txt
│   └── model/               ← paste your voc_model_export contents here
│       ├── model.pkl
│       ├── label_encoder.pkl
│       ├── top_features.pkl
│       ├── session_kmeans.pkl
│       ├── session_scaler.pkl
│       ├── session_pca.pkl
│       └── metadata.json
│
└── pi_client/
    ├── client.py            ← Runs on Raspberry Pi
    └── requirements.txt
```

---

## Step 1 — Server Setup (Your Machine)

```bash
cd server/

# paste your exported model files into server/model/
unzip voc_model_export.zip -d model/

pip install -r requirements.txt

# Start server (accessible on your local network)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Find your machine's local IP:
```bash
# Linux/Mac
ip addr | grep "inet " | grep -v 127
# Windows
ipconfig | findstr IPv4
```

Your Pi will call this IP. Example: http://192.168.1.42:8000

Verify server is running:
- Open http://localhost:8000 in browser → should see JSON
- Open http://localhost:8000/docs → Swagger UI to test manually

---

## Step 2 — Pi Setup

```bash
# On the Raspberry Pi
pip install -r requirements.txt

# Test WITHOUT hardware first (simulate=True skips sensor reading)
python client.py --server http://192.168.1.42:8000 --simulate

# With real hardware
python client.py --server http://192.168.1.42:8000 --device pi-lab-01
```

### Pi Wiring (ADS1115 ADC)

```
MQ6  analog out  → ADS1115 A0
MEMS analog out  → ADS1115 A1
ADS1115 VDD      → Pi 3.3V (pin 1)
ADS1115 GND      → Pi GND  (pin 6)
ADS1115 SDA      → Pi GPIO 2 / SDA (pin 3)
ADS1115 SCL      → Pi GPIO 3 / SCL (pin 5)
```

Enable I2C on Pi:
```bash
sudo raspi-config → Interface Options → I2C → Enable
sudo reboot
# verify: i2cdetect -y 1  (should show 0x48)
```

---

## Step 3 — Test Without Pi

Hit the API directly from your machine to confirm it works:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mq6_1_min": 3.45, "mq6_1_mean": 3.52, "mq6_1_max": 3.58,
    "mq6_1_std": 0.02, "mq6_1_median": 3.53, "mq6_1_iqr": 0.03,
    "mq6_1_skew": -0.3,
    "mems_odor_1_min": 17.4, "mems_odor_1_mean": 17.6, "mems_odor_1_max": 17.8,
    "mems_odor_1_std": 0.05, "mems_odor_1_median": 17.6, "mems_odor_1_iqr": 0.07,
    "mems_odor_1_skew": 0.1, "mems_odor_1_kurtosis": -0.5,
    "mems_odor_1_cv": 0.003, "mems_odor_1_energy": 3100.0
  }'
```

Expected response:
```json
{
  "person": "1CD24EC182",
  "confidence": 0.87,
  "status": "identified",
  "all_probs": { ... },
  "latency_ms": 3.2,
  "timestamp": "2026-02-26T..."
}
```

---

## Client Flags

| Flag          | Default              | Description                          |
|---------------|----------------------|--------------------------------------|
| `--server`    | http://localhost:8000| API server URL                       |
| `--device`    | pi-001               | Device name (logged server-side)     |
| `--window`    | 30                   | Seconds per sampling window          |
| `--rate`      | 0.1                  | Seconds between samples              |
| `--threshold` | 0.6                  | Min confidence to show as identified |
| `--simulate`  | False                | Run without hardware                 |

---

## API Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | /          | Server info + person list          |
| GET    | /health    | Health check                       |
| GET    | /persons   | List of known persons              |
| POST   | /predict   | Main inference endpoint            |
| GET    | /docs      | Swagger UI (auto-generated)        |

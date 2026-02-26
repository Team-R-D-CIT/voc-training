"""
VOC Biometric Inference + Training Server
==========================================
Raspberry Pi calls this for predictions, enrollment, and retraining.

Endpoints:
    GET  /              — server info
    GET  /health        — health check
    GET  /persons       — list of known persons
    POST /predict       — inference (existing)
    POST /enroll        — enroll new user from sensor rounds
    POST /feedback      — correction feedback from verification
    POST /retrain       — kick off background retraining
    GET  /retrain/status — poll training progress & logs

Usage:
    pip install fastapi uvicorn joblib scikit-learn pandas numpy
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
import json
import os
import csv
import time
import uuid
import copy
import shutil
import logging
import threading
from datetime import datetime
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / "model"
DATA_CSV   = BASE_DIR.parent.parent / "data.csv"        # ../../data.csv
BACKUP_DIR = MODEL_DIR / "backups"

# ── Thread-safe model container ──────────────────────────────
_model_lock = threading.RLock()


class ModelContainer:
    """Thread-safe wrapper for all model artifacts."""

    def __init__(self):
        self.model = None
        self.le = None
        self.top_features = None
        self.metadata = None
        self.features = []
        self.load()

    def load(self):
        """Load model artifacts from disk."""
        with _model_lock:
            self.model        = joblib.load(MODEL_DIR / "model.pkl")
            self.le           = joblib.load(MODEL_DIR / "label_encoder.pkl")
            self.top_features = joblib.load(MODEL_DIR / "top_features.pkl")

            with open(MODEL_DIR / "metadata.json") as f:
                self.metadata = json.load(f)

            # Detect actual feature list from the trained model
            if hasattr(self.model, 'feature_names_in_'):
                self.features = list(self.model.feature_names_in_)
            else:
                self.features = [
                    'mq6_1_min', 'mq6_1_mean', 'mq6_1_max', 'mq6_1_std',
                    'mq6_1_median', 'mq6_1_iqr', 'mq6_1_skew',
                    'mems_odor_1_min', 'mems_odor_1_mean', 'mems_odor_1_max',
                    'mems_odor_1_std', 'mems_odor_1_median', 'mems_odor_1_iqr',
                    'mems_odor_1_skew', 'mems_odor_1_kurtosis', 'mems_odor_1_cv',
                    'mems_odor_1_energy',
                    'mq6_mems_ratio', 'mq6_mems_std_ratio', 'mq6_cv',
                    'range_mq6', 'range_mems', 'skew_diff',
                    'mq6_delta', 'mems_delta', 'mq6_delta2', 'mems_delta2',
                ]

        log.info(f"Model type    : {type(self.model).__name__}")
        log.info(f"Model loaded  | {self.metadata['n_persons']} persons | {len(self.features)} features")
        log.info(f"Persons       : {self.metadata['persons']}")


# ── Init ─────────────────────────────────────────────────────
log.info("Loading model artifacts...")
try:
    mc = ModelContainer()
except Exception as e:
    log.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Model load failed: {e}")


# ── Training job state ───────────────────────────────────────
class TrainingJob:
    """Tracks the state of a background retraining job."""

    def __init__(self, job_id: str):
        self.job_id   = job_id
        self.status   = "pending"     # pending | running | done | failed
        self.logs     = []
        self.accuracy = None
        self.error    = None
        self.started  = None
        self.finished = None

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        log.info(f"[retrain {self.job_id[:8]}] {msg}")

    def to_dict(self):
        return {
            "job_id":   self.job_id,
            "status":   self.status,
            "logs":     self.logs,
            "accuracy": self.accuracy,
            "error":    self.error,
            "started":  self.started,
            "finished": self.finished,
        }


_current_job: TrainingJob | None = None
_job_lock = threading.Lock()


# ── Request / Response schemas ───────────────────────────────
class SensorReading(BaseModel):
    """Raw stats computed on the Pi from one sampling window."""
    mq6_1_min    : float
    mq6_1_mean   : float
    mq6_1_max    : float
    mq6_1_std    : float
    mq6_1_median : float
    mq6_1_iqr    : float
    mq6_1_skew   : float

    mems_odor_1_min      : float
    mems_odor_1_mean     : float
    mems_odor_1_max      : float
    mems_odor_1_std      : float
    mems_odor_1_median   : float
    mems_odor_1_iqr      : float
    mems_odor_1_skew     : float
    mems_odor_1_kurtosis : float
    mems_odor_1_cv       : float
    mems_odor_1_energy   : float

    round_no    : Optional[int]   = None
    device_id   : Optional[str]   = None
    timestamp   : Optional[str]   = None


class PredictionResponse(BaseModel):
    person        : str
    confidence    : float
    status        : str
    all_probs     : dict
    latency_ms    : float
    timestamp     : str


class EnrollRequest(BaseModel):
    """Enrollment payload from Pi: multiple rounds of sensor data."""
    user_id : str
    rounds  : List[dict]       # list of SensorReading-like dicts


class FeedbackRequest(BaseModel):
    """Correction feedback from verification."""
    predicted : str            # what the model said
    actual    : str            # what the user says is correct
    features  : dict           # the original sensor stats


class RetrainRequest(BaseModel):
    """Optional params for retraining."""
    n_estimators : Optional[int] = 300
    max_depth    : Optional[int] = None
    test_rounds  : Optional[int] = 3     # last N rounds held out for eval


# ── Feature engineering ──────────────────────────────────────
def engineer_features(row: dict) -> pd.DataFrame:
    """
    Build ALL engineered features from raw sensor stats.
    The model was trained on all 27 features (17 raw + 10 engineered).
    """
    df = pd.DataFrame([row])

    df['mq6_mems_ratio']     = df['mq6_1_mean']  / (df['mems_odor_1_mean'] + 1e-9)
    df['mq6_mems_std_ratio'] = df['mq6_1_std']   / (df['mems_odor_1_std']  + 1e-9)
    df['mq6_cv']             = df['mq6_1_std']   / (df['mq6_1_mean']       + 1e-9)
    df['range_mq6']          = df['mq6_1_max']   - df['mq6_1_min']
    df['range_mems']         = df['mems_odor_1_max'] - df['mems_odor_1_min']
    df['skew_diff']          = df['mq6_1_skew']  - df['mems_odor_1_skew']
    df['mq6_delta']  = 0
    df['mems_delta'] = 0
    df['mq6_delta2'] = 0
    df['mems_delta2'] = 0

    for feat in mc.features:
        if feat not in df.columns:
            df[feat] = 0

    return df[mc.features].fillna(0)


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for a full DataFrame (used during retraining).
    Mirrors the notebook Step 4 exactly.
    """
    df = df.copy()
    df['mq6_mems_ratio']     = df['mq6_1_mean']  / (df['mems_odor_1_mean'] + 1e-9)
    df['mq6_mems_std_ratio'] = df['mq6_1_std']   / (df['mems_odor_1_std']  + 1e-9)
    df['mq6_cv']             = df['mq6_1_std']   / (df['mq6_1_mean']       + 1e-9)
    df['range_mq6']          = df['mq6_1_max']   - df['mq6_1_min']
    df['range_mems']         = df['mems_odor_1_max'] - df['mems_odor_1_min']
    df['skew_diff']          = df['mq6_1_skew']  - df['mems_odor_1_skew']

    # Delta features per person
    df['mq6_delta']  = df.groupby('user_id')['mq6_1_mean'].diff().fillna(0)
    df['mems_delta'] = df.groupby('user_id')['mems_odor_1_mean'].diff().fillna(0)
    df['mq6_delta2'] = df.groupby('user_id')['mq6_delta'].diff().fillna(0)
    df['mems_delta2'] = df.groupby('user_id')['mems_delta'].diff().fillna(0)

    return df


# ── CSV helpers ──────────────────────────────────────────────
CSV_COLUMNS = None   # set dynamically from data.csv header


def _read_csv_columns():
    """Read the column names from data.csv."""
    global CSV_COLUMNS
    if DATA_CSV.exists():
        with open(DATA_CSV) as f:
            reader = csv.reader(f)
            CSV_COLUMNS = next(reader)
    return CSV_COLUMNS


def _next_id():
    """Get the next row id for data.csv."""
    if not DATA_CSV.exists():
        return 1
    df = pd.read_csv(DATA_CSV)
    return int(df['id'].max()) + 1 if len(df) > 0 else 1


def _append_rows_to_csv(rows: list[dict]):
    """Append rows to data.csv, creating it if needed."""
    cols = _read_csv_columns()
    if cols is None:
        raise RuntimeError("data.csv not found — cannot append")

    with open(DATA_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        for row in rows:
            clean = {}
            for c in cols:
                clean[c] = row.get(c, 0)
            writer.writerow(clean)


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="VOC Biometric API",
    description="Person identification, enrollment & retraining from VOC sensors",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIDENCE_THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════════
#  EXISTING ROUTES (unchanged)
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    with _model_lock:
        return {
            "service"   : "VOC Biometric API",
            "status"    : "running",
            "model"     : type(mc.model).__name__,
            "persons"   : mc.metadata["persons"],
            "n_features": len(mc.features),
        }


@app.get("/health")
def health():
    with _model_lock:
        return {
            "status": "ok",
            "model": type(mc.model).__name__,
            "n_persons": mc.metadata["n_persons"],
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/persons")
def list_persons():
    with _model_lock:
        return {"persons": mc.metadata["persons"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    t_start = time.perf_counter()

    try:
        X = engineer_features(reading.dict())
        X_np = X.values

        with _model_lock:
            local_model = mc.model
            local_le    = mc.le

        if hasattr(local_model, "predict_proba"):
            probs      = local_model.predict_proba(X_np)[0]
            pred_idx   = int(probs.argmax())
            confidence = float(probs[pred_idx])
            all_probs  = {str(cls): round(float(p), 4)
                          for cls, p in zip(local_le.classes_, probs)}
        else:
            pred_idx   = int(local_model.predict(X_np)[0])
            confidence = 1.0
            all_probs  = {}

        person = str(local_le.inverse_transform([pred_idx])[0])
        status = "identified" if confidence >= CONFIDENCE_THRESHOLD else "uncertain"

    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t_start) * 1000

    log.info(
        f"[{reading.device_id or 'unknown'}] "
        f"→ {person} ({confidence:.3f}) | {latency_ms:.1f}ms | {status}"
    )

    return PredictionResponse(
        person     = person,
        confidence = round(confidence, 4),
        status     = status,
        all_probs  = all_probs,
        latency_ms = round(latency_ms, 2),
        timestamp  = datetime.now().isoformat(),
    )


# ══════════════════════════════════════════════════════════════
#  NEW: ENROLLMENT
# ══════════════════════════════════════════════════════════════

@app.post("/enroll")
def enroll(req: EnrollRequest):
    """
    Receive enrollment rounds from Pi and append to data.csv.
    Each round is a dict of sensor stats (same shape as SensorReading).
    """
    if not req.user_id or not req.user_id.strip():
        raise HTTPException(400, "user_id is required")
    if not req.rounds or len(req.rounds) == 0:
        raise HTTPException(400, "At least one round of data required")

    user_id = req.user_id.strip()
    next_id = _next_id()

    # Determine the next round_no for this user
    if DATA_CSV.exists():
        df_existing = pd.read_csv(DATA_CSV)
        user_rows = df_existing[df_existing['user_id'] == user_id]
        start_round = int(user_rows['round_no'].max()) + 1 if len(user_rows) > 0 else 1
    else:
        start_round = 1

    rows_to_add = []
    for i, rnd in enumerate(req.rounds):
        row = {
            "id":       next_id + i,
            "user_id":  user_id,
            "round_no": start_round + i,
        }
        # Copy all sensor stat fields
        for key, val in rnd.items():
            if key not in ("round_no", "device_id", "timestamp"):
                row[key] = val
        rows_to_add.append(row)

    try:
        _append_rows_to_csv(rows_to_add)
    except Exception as e:
        log.error(f"Enrollment CSV write failed: {e}")
        raise HTTPException(500, f"Failed to save enrollment data: {e}")

    log.info(f"✅ Enrolled {user_id}: {len(rows_to_add)} rounds (rounds {start_round}–{start_round + len(rows_to_add) - 1})")

    return {
        "status":     "enrolled",
        "user_id":    user_id,
        "n_rounds":   len(rows_to_add),
        "round_range": [start_round, start_round + len(rows_to_add) - 1],
    }


# ══════════════════════════════════════════════════════════════
#  NEW: FEEDBACK
# ══════════════════════════════════════════════════════════════

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """
    Record a correction: model predicted X but the true person is Y.
    Appends the corrected sample to data.csv with the true label.
    """
    if not req.actual or not req.actual.strip():
        raise HTTPException(400, "actual person label is required")

    next_id = _next_id()

    row = {
        "id":       next_id,
        "user_id":  req.actual.strip(),
        "round_no": 99,   # flag feedback rows with a special round number
    }
    for key, val in req.features.items():
        if key not in ("round_no", "device_id", "timestamp"):
            row[key] = val

    try:
        _append_rows_to_csv([row])
    except Exception as e:
        log.error(f"Feedback CSV write failed: {e}")
        raise HTTPException(500, f"Failed to save feedback: {e}")

    log.info(f"📝 Feedback: predicted={req.predicted} → actual={req.actual}")

    return {
        "status":    "feedback_recorded",
        "predicted": req.predicted,
        "actual":    req.actual,
    }


# ══════════════════════════════════════════════════════════════
#  NEW: RETRAIN
# ══════════════════════════════════════════════════════════════

def _run_retraining(job: TrainingJob, params: RetrainRequest):
    """Background retraining thread."""
    global mc

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    job.status  = "running"
    job.started = datetime.now().isoformat()
    t0 = time.time()

    try:
        # ── 1. Load data ─────────────────────────────────────
        job.log(f"Loading data from {DATA_CSV}")
        if not DATA_CSV.exists():
            raise FileNotFoundError(f"Training data not found: {DATA_CSV}")

        df = pd.read_csv(DATA_CSV)
        job.log(f"Loaded {len(df)} rows, {df['user_id'].nunique()} persons")

        # Show per-person counts
        counts = df['user_id'].value_counts()
        for uid, cnt in counts.items():
            job.log(f"  {uid}: {cnt} samples")

        # ── 2. Feature engineering ────────────────────────────
        job.log("Engineering features…")
        df = engineer_features_df(df)

        # Define feature columns (exclude id, user_id, round_no)
        raw_sensor_cols = [c for c in df.columns
                          if c.startswith(('mq6_1_', 'mems_odor_1_'))
                          and c in df.columns]
        engineered_cols = [
            'mq6_mems_ratio', 'mq6_mems_std_ratio', 'mq6_cv',
            'range_mq6', 'range_mems', 'skew_diff',
            'mq6_delta', 'mems_delta', 'mq6_delta2', 'mems_delta2',
        ]
        feature_cols = raw_sensor_cols + engineered_cols
        feature_cols = [c for c in feature_cols if c in df.columns]

        job.log(f"Using {len(feature_cols)} features: {feature_cols[:10]}…")

        # ── 3. Train/test split by round ──────────────────────
        max_round = int(df['round_no'].max())
        test_cutoff = max_round - params.test_rounds + 1
        train_mask = df['round_no'] < test_cutoff
        test_mask  = df['round_no'] >= test_cutoff

        X_train = df.loc[train_mask, feature_cols].fillna(0)
        X_test  = df.loc[test_mask,  feature_cols].fillna(0)

        le_new = LabelEncoder()
        y_all = le_new.fit_transform(df['user_id'])
        y_train = y_all[train_mask.values]
        y_test  = y_all[test_mask.values]

        job.log(f"Train: {len(X_train)} rows (rounds 1–{test_cutoff - 1})")
        job.log(f"Test : {len(X_test)} rows (rounds {test_cutoff}–{max_round})")
        job.log(f"Persons: {len(le_new.classes_)}")

        if len(X_train) < 5:
            raise ValueError("Not enough training data")

        # ── 4. Cross-validation ───────────────────────────────
        job.log(f"Training RandomForestClassifier (n_estimators={params.n_estimators})…")
        new_model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_leaf=1,
            min_samples_split=2,
            class_weight='balanced',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        n_splits = min(5, len(set(y_train)))
        if n_splits >= 2 and len(X_train) >= n_splits:
            job.log(f"Running {n_splits}-fold cross-validation on training set…")
            cv_scores = cross_val_score(new_model, X_train, y_train,
                                        cv=n_splits, scoring='accuracy')
            for i, s in enumerate(cv_scores):
                job.log(f"  Fold {i+1}: {s:.4f}")
            job.log(f"  CV mean accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── 5. Final fit ──────────────────────────────────────
        job.log("Fitting final model on full training set…")
        new_model.fit(X_train, y_train)
        job.log(f"Training complete. {new_model.n_estimators} trees built.")

        # ── 6. Evaluate on test set ───────────────────────────
        if len(X_test) > 0:
            test_acc = float(new_model.score(X_test, y_test))
            job.log(f"✅ Test accuracy: {test_acc:.4f} ({int(test_acc * len(y_test))}/{len(y_test)} correct)")
            job.accuracy = round(test_acc, 4)

            # Per-person test accuracy
            preds = new_model.predict(X_test)
            for cls_idx, cls_name in enumerate(le_new.classes_):
                mask = (y_test == cls_idx)
                if mask.sum() > 0:
                    cls_acc = (preds[mask] == y_test[mask]).mean()
                    job.log(f"  {cls_name}: {cls_acc:.2%} ({mask.sum()} test samples)")
        else:
            job.log("⚠ No test data — skipping evaluation")
            job.accuracy = None

        # ── 7. Feature importance ─────────────────────────────
        importances = new_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        top_feat_names = [feature_cols[i] for i in top_idx]
        job.log(f"Top features: {top_feat_names[:5]}…")

        # ── 8. Save artifacts ─────────────────────────────────
        job.log("Saving model artifacts…")

        # Backup old model
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for f in MODEL_DIR.glob("*.pkl"):
            shutil.copy2(f, BACKUP_DIR / f"{f.stem}_{ts}{f.suffix}")
        for f in MODEL_DIR.glob("*.json"):
            shutil.copy2(f, BACKUP_DIR / f"{f.stem}_{ts}{f.suffix}")
        job.log(f"Old model backed up to {BACKUP_DIR}")

        # Save new artifacts
        joblib.dump(new_model,      MODEL_DIR / "model.pkl")
        joblib.dump(le_new,         MODEL_DIR / "label_encoder.pkl")
        joblib.dump(top_feat_names, MODEL_DIR / "top_features.pkl")

        new_metadata = {
            "n_persons":     len(le_new.classes_),
            "n_features":    len(feature_cols),
            "top_features":  top_feat_names,
            "persons":       list(le_new.classes_),
            "max_round":     max_round,
            "train_cutoff":  test_cutoff,
            "trained_at":    datetime.now().isoformat(),
            "test_accuracy": job.accuracy,
        }
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(new_metadata, f, indent=2)

        job.log("Artifacts saved to disk.")

        # ── 9. Hot-reload model in memory ─────────────────────
        job.log("Hot-reloading model in memory…")
        mc.load()
        job.log(f"✅ Model reloaded: {type(mc.model).__name__}, {len(mc.features)} features, {mc.metadata['n_persons']} persons")

        # ── Done ──────────────────────────────────────────────
        elapsed = time.time() - t0
        job.log(f"✅ Retraining complete in {elapsed:.1f}s")
        job.status   = "done"
        job.finished = datetime.now().isoformat()

    except Exception as e:
        job.log(f"❌ Retraining failed: {e}")
        job.status   = "failed"
        job.error    = str(e)
        job.finished = datetime.now().isoformat()
        log.error(f"Retraining failed: {e}", exc_info=True)


@app.post("/retrain")
def retrain(params: RetrainRequest = RetrainRequest()):
    """Kick off model retraining in a background thread."""
    global _current_job

    with _job_lock:
        if _current_job and _current_job.status == "running":
            raise HTTPException(409, "A retraining job is already running")

        job_id = str(uuid.uuid4())[:12]
        _current_job = TrainingJob(job_id)

    thread = threading.Thread(
        target=_run_retraining,
        args=(_current_job, params),
        daemon=True,
    )
    thread.start()

    log.info(f"🔄 Retraining started (job {job_id})")

    return {
        "status":  "training_started",
        "job_id":  job_id,
    }


@app.get("/retrain/status")
def retrain_status():
    """Poll the current retraining job status."""
    with _job_lock:
        if _current_job is None:
            return {"status": "no_job", "logs": [], "accuracy": None}
        return _current_job.to_dict()


@app.get("/retrain/logs/stream")
def retrain_logs_stream():
    """
    SSE stream of retraining logs.
    The client reads this as text/event-stream line by line.
    """
    def event_generator():
        last_idx = 0
        while True:
            with _job_lock:
                job = _current_job

            if job is None:
                yield f"data: {json.dumps({'type': 'info', 'msg': 'No active job'})}\n\n"
                return

            # Send any new log lines
            current_logs = job.logs[last_idx:]
            for line in current_logs:
                yield f"data: {json.dumps({'type': 'log', 'msg': line})}\n\n"
            last_idx = len(job.logs)

            # Check if done
            if job.status in ("done", "failed"):
                yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'accuracy': job.accuracy, 'error': job.error})}\n\n"
                return

            time.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

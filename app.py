# ═══════════════════════════════════════════════════════════════
# FireCast Backend — FastAPI + APScheduler
# Run: uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs
# ═══════════════════════════════════════════════════════════════

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from gridmet_service import fetch_gridmet_window, test_gridmet_connection

from database import (
    SessionLocal, Subscriber,
    AlertHistory, SchedulerLog, init_db
)
from email_service import send_fire_alert_email, send_test_email

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
MODEL_PATH = "model/best_model_5.0x.h5"
PRED_CSV   = "data/all_predictions.csv"

# ── Global State ───────────────────────────────────────────────
model          = None
predictions_df = None
scheduler      = None

# ── Region Bounds ──────────────────────────────────────────────
REGION_BOUNDS = {
    'california': (32.5, 42.0, -124.5, -114.0),
    'pacific_nw': (42.0, 49.0, -124.5, -110.0),
    'southeast':  (25.0, 36.0, -95.0,  -75.0),
    'southwest':  (31.0, 42.0, -117.0, -102.0),
    'rockies':    (36.0, 49.0, -116.0, -102.0),
}


# ══════════════════════════════════════════════════════════════
# SCHEDULER JOB — runs every day at 6AM
# ══════════════════════════════════════════════════════════════
async def run_daily_alerts():
    """
    Main scheduler job — runs automatically every day at 6AM.
    Checks all subscribers and sends alerts if risk is high.
    """
    logger.info("=" * 55)
    logger.info("⏰ SCHEDULER STARTED — Daily Alert Check")
    logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 55)

    db             = SessionLocal()
    alerts_sent    = 0
    errors         = 0
    checked        = 0

    # ── Create scheduler log entry ─────────────────────────────
    log = SchedulerLog(
        run_date = datetime.utcnow(),
        status   = "running"
    )
    db.add(log)
    db.commit()

    try:
        # ── Get all active subscribers ─────────────────────────
        subscribers = db.query(Subscriber).filter(
            Subscriber.active == True
        ).all()

        logger.info(
            f"📋 Found {len(subscribers)} active subscribers"
        )

        for sub in subscribers:
            try:
                checked += 1
                logger.info(
                    f"  Checking: {sub.email} "
                    f"({sub.location_name}) "
                    f"threshold={sub.threshold}"
                )

                # ── Real prediction from CSV ───────────────────
                location_data = get_prediction_for_location(
                    sub.latitude,
                    sub.longitude
                )

                if location_data is None:
                    logger.warning(
                        f"  ⚠️ No data for "
                        f"({sub.latitude}, {sub.longitude})"
                    )
                    errors += 1
                    continue

                prob       = location_data['fire_prob']
                risk_level, color, _, message = get_risk_info(prob)

                # ── Calibrated threshold ───────────────────────
                calibrated = location_data['calibrated_threshold']
                effective  = max(sub.threshold, calibrated)

                logger.info(
                    f"  prob={prob:.4f} "
                    f"effective_threshold={effective:.2f} "
                    f"zone={location_data['fire_zone']}"
                )

                # ── Check if alert needed ──────────────────────
                if prob >= effective:

                    # ── Check: don't send same alert twice ─────
                    today = datetime.utcnow().date()
                    already_sent = db.query(AlertHistory).filter(
                        AlertHistory.email     == sub.email,
                        AlertHistory.alert_sent == True
                    ).filter(
                        AlertHistory.sent_at >= datetime(
                            today.year,
                            today.month,
                            today.day
                        )
                    ).first()

                    if already_sent:
                        logger.info(
                            f"  ⏭️  Already sent today — skipping"
                        )
                        continue

                    # ── Send alert email ───────────────────────
                    email_sent = send_fire_alert_email(
                        to_email      = sub.email,
                        location_name = sub.location_name,
                        latitude      = sub.latitude,
                        longitude     = sub.longitude,
                        fire_prob     = prob,
                        risk_level    = risk_level,
                        extra_info    = {
                            'data_date':  location_data['date'],
                            'peak_prob':  location_data['peak_fire_prob'],
                            'peak_date':  location_data['peak_date'],
                            'match_dist': location_data['distance_deg'],
                            'fire_zone':  location_data['fire_zone'],
                            'true_fire_rate': location_data['true_fire_rate']
                        }
                    )

                    if email_sent:
                        alerts_sent += 1
                        logger.info(
                            f"  ✅ Alert sent to {sub.email}!"
                        )
                    else:
                        errors += 1
                        logger.error(
                            f"  ❌ Email failed for {sub.email}"
                        )

                    # ── Log to database ────────────────────────
                    alert_log = AlertHistory(
                        email      = sub.email,
                        latitude   = sub.latitude,
                        longitude  = sub.longitude,
                        fire_prob  = prob,
                        risk_level = risk_level,
                        alert_sent = email_sent
                    )
                    db.add(alert_log)
                    db.commit()

                else:
                    logger.info(
                        f"  ✅ Risk below threshold — no alert"
                    )

            except Exception as e:
                errors += 1
                logger.error(
                    f"  ❌ Error for {sub.email}: {e}"
                )
                continue

        # ── Update scheduler log ───────────────────────────────
        log.subscribers_checked = checked
        log.alerts_sent         = alerts_sent
        log.errors              = errors
        log.status              = "completed"
        log.finished_at         = datetime.utcnow()
        db.commit()

        logger.info("=" * 55)
        logger.info("✅ SCHEDULER COMPLETED")
        logger.info(f"   Checked:  {checked} subscribers")
        logger.info(f"   Alerts:   {alerts_sent} sent")
        logger.info(f"   Errors:   {errors}")
        logger.info("=" * 55)

    except Exception as e:
        log.status      = "failed"
        log.errors      = errors
        log.finished_at = datetime.utcnow()
        db.commit()
        logger.error(f"❌ Scheduler failed: {e}")

    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# LIFESPAN — startup + shutdown
# ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, predictions_df, scheduler

    logger.info("=" * 55)
    logger.info("🔥 FIRECAST BACKEND STARTING...")
    logger.info("=" * 55)

    # Init database
    init_db()

    # Load model
    logger.info("Loading ConvLSTM2D model...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    logger.info(f"✅ Model loaded! Params: {model.count_params():,}")

    # Load predictions CSV
    logger.info("Loading predictions CSV...")
    predictions_df = pd.read_csv(PRED_CSV)
    predictions_df['date']  = pd.to_datetime(predictions_df['date'])
    predictions_df['year']  = predictions_df['date'].dt.year
    predictions_df['month'] = predictions_df['date'].dt.month
    logger.info(f"✅ Loaded {len(predictions_df):,} predictions")

    # ── Start Scheduler ────────────────────────────────────────
    scheduler = AsyncIOScheduler()

    # Run every day at 6:00 AM
    scheduler.add_job(
        run_daily_alerts,
        CronTrigger(hour=6, minute=0),
        id='daily_alerts',
        name='Daily Fire Alert Check',
        replace_existing=True
    )

    scheduler.start()
    logger.info("✅ Scheduler started → runs daily at 6:00 AM")

    logger.info("=" * 55)
    logger.info("✅ ALL SYSTEMS READY!")
    logger.info("   Website: http://localhost:8000")
    logger.info("   Docs:    http://localhost:8000/docs")
    logger.info("=" * 55)

    yield

    # ── Shutdown ───────────────────────────────────────────────
    logger.info("🛑 Shutting down FireCast...")
    scheduler.shutdown()
    del model
    del predictions_df
    tf.keras.backend.clear_session()


# ══════════════════════════════════════════════════════════════
# APP INIT
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title="🔥 FireCast API",
    description="""
## Wildfire Prediction & Trajectory System

Built with **ConvLSTM2D** deep learning model trained on
6.7M weather samples across 36,854 USA locations (2014–2025).

### Model Performance
- **AUC**: 0.9082
- **Recall**: 99.4% (catches 99/100 real fires)
- **Training**: 31 epochs on Google Colab T4 GPU

### Scheduler
- Runs automatically every day at **6:00 AM**
- Checks all subscribers
- Sends email alerts if risk crosses threshold
    """,
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    temperature:   float = Field(..., ge=-20, le=60,  example=42.0)
    humidity:      float = Field(..., ge=0,   le=100, example=12.0)
    wind_speed:    float = Field(..., ge=0,   le=50,  example=8.5)
    precipitation: float = Field(..., ge=0,   le=200, example=0.0)
    dry_days:      float = Field(..., ge=0,   le=75,  example=45.0)
    erc:           float = Field(..., ge=0,   le=150, example=72.0)


class PredictResponse(BaseModel):
    fire_probability: float
    fire_percentage:  float
    risk_level:       str
    color:            str
    alert:            bool
    message:          str
    inputs:           dict


class SubscribeRequest(BaseModel):
    email:         str   = Field(...,          example="your@gmail.com")
    latitude:      float = Field(...,          example=37.7749)
    longitude:     float = Field(...,          example=-119.4194)
    location_name: str   = Field(default="Custom Location",
                                 example="California")
    threshold:     float = Field(default=0.7, example=0.7)


class TestAlertRequest(BaseModel):
    email:         str   = Field(...,          example="your@gmail.com")
    latitude:      float = Field(...,          example=37.7749)
    longitude:     float = Field(...,          example=-119.4194)
    location_name: str   = Field(default="Test Location",
                                 example="California")
    threshold:     float = Field(default=0.5,  example=0.5)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def get_risk_info(prob: float):
    if prob >= 0.90:
        return "EXTREME", "#ff2200", True, (
            "⚠️ EXTREME fire risk! Conditions are critical."
        )
    elif prob >= 0.70:
        return "HIGH", "#ff6600", True, (
            "🔶 HIGH fire risk detected."
        )
    elif prob >= 0.50:
        return "MODERATE", "#ffaa00", False, (
            "🟡 Moderate fire risk. Stay aware."
        )
    else:
        return "LOW", "#00ff88", False, (
            "✅ Low fire risk. Conditions are safe."
        )


def get_prediction_for_location(lat: float, lng: float):
    try:
        predictions_df['dist'] = (
            (predictions_df['latitude']  - lat).abs() +
            (predictions_df['longitude'] - lng).abs()
        )
        closest = predictions_df.nsmallest(200, 'dist').copy()

        if len(closest) == 0:
            return None

        latest = closest.sort_values(
            'date', ascending=False
        ).iloc[0]
        peak   = closest.nlargest(1, 'fire_prob').iloc[0]

        closest['month']  = closest['date'].dt.month
        monthly_avg       = closest.groupby('month')['fire_prob']\
                                   .mean().to_dict()
        true_fire_rate    = float(closest['true_fire'].mean())

        # Calibrated threshold based on fire zone
        if true_fire_rate >= 0.10:
            calibrated_threshold = 0.70
            zone = "HIGH FIRE ZONE"
        elif true_fire_rate >= 0.05:
            calibrated_threshold = 0.80
            zone = "MEDIUM FIRE ZONE"
        elif true_fire_rate >= 0.02:
            calibrated_threshold = 0.92
            zone = "LOW FIRE ZONE"
        else:
            calibrated_threshold = 0.95
            zone = "VERY LOW FIRE ZONE"

        # Adjusted probability
        raw_prob      = float(latest['fire_prob'])
        adjusted_prob = (raw_prob * 0.6) + (true_fire_rate * 10 * 0.4)
        adjusted_prob = min(max(adjusted_prob, 0.0), 1.0)

        predictions_df.drop(
            columns=['dist'], inplace=True, errors='ignore'
        )

        return {
            'fire_prob':            round(adjusted_prob,   4),
            'raw_prob':             round(raw_prob,        4),
            'true_fire_rate':       round(true_fire_rate,  4),
            'calibrated_threshold': calibrated_threshold,
            'fire_zone':            zone,
            'date':                 str(latest['date'].date()),
            'true_fire':            int(latest['true_fire']),
            'matched_lat':          round(float(latest['latitude']),  4),
            'matched_lng':          round(float(latest['longitude']), 4),
            'distance_deg':         round(float(
                abs(latest['latitude']  - lat) +
                abs(latest['longitude'] - lng)
            ), 4),
            'peak_fire_prob':       round(float(peak['fire_prob']), 4),
            'peak_date':            str(peak['date'].date()),
            'total_samples':        len(closest),
            'monthly_avg':          {
                int(k): round(float(v), 4)
                for k, v in monthly_avg.items()
            }
        }

    except Exception as e:
        logger.error(f"Location lookup error: {e}")
        predictions_df.drop(
            columns=['dist'], inplace=True, errors='ignore'
        )
        return None


def build_window(temperature, humidity, wind_speed,
                 precipitation, dry_days, erc):
    window = []
    for day in range(75):
        progress = day / 74.0
        row = [
            0.0, 0.0,
            precipitation * (1 - progress),
            80 - (humidity * progress * 0.5),
            humidity + (15 * (1 - progress)),
            0.008,
            200 + (100 * progress),
            temperature - 12 + (6  * progress),
            temperature - 6  + (8  * progress),
            wind_speed * (0.3 + 0.7 * progress),
            erc * progress,
            30 - (15 * progress),
            45 - (20 * progress),
            erc * progress,
            3.5, 2.8,
            (temperature * 0.08) * progress
        ]
        window.append(row)
    x = np.array(window, dtype=np.float32)
    x = x[np.newaxis, :, np.newaxis, np.newaxis, :]
    return x


# ══════════════════════════════════════════════════════════════
# STATIC FILES
# ══════════════════════════════════════════════════════════════
app.mount(
    "/trajectory_maps",
    StaticFiles(directory="static/trajectory_maps"),
    name="trajectory_maps"
)

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("static/index.html")


# ══════════════════════════════════════════════════════════════
# ROUTE 1 — Health Check
# ══════════════════════════════════════════════════════════════
@app.get("/api/health", summary="Health Check", tags=["System"])
async def health():
    db   = SessionLocal()
    subs = db.query(Subscriber).filter(
        Subscriber.active == True
    ).count()
    db.close()

    next_run = None
    if scheduler:
        job = scheduler.get_job('daily_alerts')
        if job and job.next_run_time:
            next_run = str(job.next_run_time)

    return {
        "status":               "ok",
        "model_loaded":         model is not None,
        "data_loaded":          predictions_df is not None,
        "total_samples":        len(predictions_df) if predictions_df is not None else 0,
        "tensorflow":           tf.__version__,
        "active_subscribers":   subs,
        "scheduler_running":    scheduler.running if scheduler else False,
        "next_scheduled_run":   next_run
    }


# ══════════════════════════════════════════════════════════════
# ROUTE 2 — Predict Fire Risk
# ══════════════════════════════════════════════════════════════
@app.post(
    "/api/predict",
    response_model=PredictResponse,
    summary="Predict Fire Risk",
    tags=["Prediction"]
)
async def predict(body: PredictRequest):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        x    = build_window(
            body.temperature, body.humidity,
            body.wind_speed,  body.precipitation,
            body.dry_days,    body.erc
        )
        prob = float(
            model(tf.constant(x), training=False).numpy()[0][0]
        )
        risk_level, color, alert, message = get_risk_info(prob)

        return PredictResponse(
            fire_probability = round(prob, 4),
            fire_percentage  = round(prob * 100, 1),
            risk_level       = risk_level,
            color            = color,
            alert            = alert,
            message          = message,
            inputs           = body.model_dump()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# ROUTE 3 — Subscribe
# ══════════════════════════════════════════════════════════════
@app.post("/api/subscribe", summary="Subscribe to Alerts", tags=["Alerts"])
async def subscribe(body: SubscribeRequest):
    db = SessionLocal()
    try:
        existing = db.query(Subscriber).filter(
            Subscriber.email == body.email
        ).first()

        if existing:
            existing.latitude      = body.latitude
            existing.longitude     = body.longitude
            existing.location_name = body.location_name
            existing.threshold     = body.threshold
            existing.active        = True
            db.commit()
            return {
                "status":  "updated",
                "message": f"✅ Subscription updated for {body.email}"
            }

        sub = Subscriber(
            email         = body.email,
            latitude      = body.latitude,
            longitude     = body.longitude,
            location_name = body.location_name,
            threshold     = body.threshold
        )
        db.add(sub)
        db.commit()
        send_test_email(body.email)

        return {
            "status":    "subscribed",
            "message":   f"✅ Subscribed! Check {body.email} for confirmation.",
            "threshold": f"{round(body.threshold * 100)}%",
            "scheduler": "Alerts run automatically every day at 6:00 AM ✅"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# ROUTE 4 — Test Alert
# ══════════════════════════════════════════════════════════════
@app.post("/api/test-alert", summary="Send Test Alert", tags=["Alerts"])
async def test_alert(body: TestAlertRequest):
    try:
        location_data = get_prediction_for_location(
            body.latitude, body.longitude
        )

        if location_data is None:
            raise HTTPException(
                status_code=404,
                detail="No data for this location. Must be within USA."
            )

        prob       = location_data['fire_prob']
        risk_level, color, _, message = get_risk_info(prob)

        calibrated        = location_data['calibrated_threshold']
        effective         = max(body.threshold, calibrated)
        crosses_threshold = prob >= effective
        email_sent        = False

        if crosses_threshold:
            email_sent = send_fire_alert_email(
                to_email      = body.email,
                location_name = body.location_name,
                latitude      = body.latitude,
                longitude     = body.longitude,
                fire_prob     = prob,
                risk_level    = risk_level,
                extra_info    = {
                    'data_date':      location_data['date'],
                    'peak_prob':      location_data['peak_fire_prob'],
                    'peak_date':      location_data['peak_date'],
                    'match_dist':     location_data['distance_deg'],
                    'fire_zone':      location_data['fire_zone'],
                    'true_fire_rate': location_data['true_fire_rate']
                }
            )

            db = SessionLocal()
            try:
                db.add(AlertHistory(
                    email      = body.email,
                    latitude   = body.latitude,
                    longitude  = body.longitude,
                    fire_prob  = prob,
                    risk_level = risk_level,
                    alert_sent = email_sent
                ))
                db.commit()
            finally:
                db.close()

        return {
            "fire_probability":    round(prob, 4),
            "raw_probability":     location_data['raw_prob'],
            "fire_percentage":     round(prob * 100, 1),
            "risk_level":          risk_level,
            "color":               color,
            "threshold":           body.threshold,
            "effective_threshold": effective,
            "crosses_threshold":   crosses_threshold,
            "email_sent":          email_sent,
            "fire_zone":           location_data['fire_zone'],
            "true_fire_rate":      location_data['true_fire_rate'],
            "data_source": {
                "data_date":    location_data['date'],
                "matched_lat":  location_data['matched_lat'],
                "matched_lng":  location_data['matched_lng'],
                "distance_km":  round(location_data['distance_deg'] * 111, 1),
                "peak_prob":    location_data['peak_fire_prob'],
                "peak_date":    location_data['peak_date']
            },
            "message": (
                f"✅ Real alert sent to {body.email}! "
                f"Zone: {location_data['fire_zone']} | "
                f"True fire rate: {round(location_data['true_fire_rate']*100,1)}%"
                if email_sent else
                f"ℹ️ Risk {round(prob*100,1)}% is below "
                f"effective threshold {round(effective*100)}% "
                f"({location_data['fire_zone']}). No alert needed."
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# ROUTE 5 — Manually Trigger Scheduler (for testing!)
# POST /api/run-scheduler
# ══════════════════════════════════════════════════════════════
@app.post(
    "/api/run-scheduler",
    summary="Manually Trigger Daily Scheduler",
    tags=["Scheduler"]
)
async def run_scheduler_now():
    """
    Manually triggers the daily alert scheduler RIGHT NOW.
    Perfect for testing without waiting until 6AM!
    Checks all subscribers and sends alerts automatically.
    """
    logger.info("🔧 Manual scheduler trigger via API")
    await run_daily_alerts()
    return {
        "status":  "completed",
        "message": "✅ Scheduler ran successfully! Check logs and alert history.",
        "time":    str(datetime.now())
    }


# ══════════════════════════════════════════════════════════════
# ROUTE 6 — Scheduler Status
# GET /api/scheduler-status
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/scheduler-status",
    summary="Check Scheduler Status",
    tags=["Scheduler"]
)
async def scheduler_status():
    """
    Returns scheduler status, next run time,
    and history of all past runs.
    """
    db = SessionLocal()
    try:
        logs = db.query(SchedulerLog).order_by(
            SchedulerLog.run_date.desc()
        ).limit(10).all()

        next_run = None
        if scheduler:
            job = scheduler.get_job('daily_alerts')
            if job and job.next_run_time:
                next_run = str(job.next_run_time)

        return {
            "scheduler_running": scheduler.running if scheduler else False,
            "next_run_time":     next_run,
            "schedule":          "Every day at 6:00 AM",
            "past_runs": [
                {
                    "run_date":            str(l.run_date),
                    "status":              l.status,
                    "subscribers_checked": l.subscribers_checked,
                    "alerts_sent":         l.alerts_sent,
                    "errors":              l.errors,
                    "finished_at":         str(l.finished_at)
                }
                for l in logs
            ]
        }
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# ROUTE 7 — Unsubscribe
# ══════════════════════════════════════════════════════════════
@app.delete(
    "/api/unsubscribe/{email}",
    summary="Unsubscribe from Alerts",
    tags=["Alerts"]
)
async def unsubscribe(email: str):
    db = SessionLocal()
    try:
        sub = db.query(Subscriber).filter(
            Subscriber.email == email
        ).first()
        if not sub:
            raise HTTPException(status_code=404, detail="Email not found")
        sub.active = False
        db.commit()
        return {"status": "unsubscribed", "message": f"✅ {email} unsubscribed"}
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# ROUTE 8 — Alert History
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/alert-history",
    summary="View Alert History",
    tags=["Alerts"]
)
async def alert_history(
    email: Optional[str] = Query(None),
    limit: int           = Query(20, ge=1, le=100)
):
    db = SessionLocal()
    try:
        query = db.query(AlertHistory)
        if email:
            query = query.filter(AlertHistory.email == email)
        logs = query.order_by(
            AlertHistory.sent_at.desc()
        ).limit(limit).all()

        return {
            "count": len(logs),
            "alerts": [
                {
                    "email":      l.email,
                    "latitude":   l.latitude,
                    "longitude":  l.longitude,
                    "fire_prob":  round(l.fire_prob, 4),
                    "risk_level": l.risk_level,
                    "email_sent": l.alert_sent,
                    "sent_at":    str(l.sent_at)
                }
                for l in logs
            ]
        }
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# ROUTE 9 — Get All Subscribers
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/subscribers",
    summary="Get All Subscribers",
    tags=["Alerts"]
)
async def get_subscribers():
    db = SessionLocal()
    try:
        subs = db.query(Subscriber).filter(
            Subscriber.active == True
        ).all()
        return {
            "count": len(subs),
            "subscribers": [
                {
                    "email":         s.email,
                    "location_name": s.location_name,
                    "latitude":      s.latitude,
                    "longitude":     s.longitude,
                    "threshold":     s.threshold,
                    "created_at":    str(s.created_at)
                }
                for s in subs
            ]
        }
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# ROUTE 10 — Location Lookup
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/location-lookup",
    summary="Get Real Prediction For Any Location",
    tags=["Prediction"]
)
async def location_lookup(
    lat: float = Query(..., example=37.7749),
    lng: float = Query(..., example=-119.4194)
):
    if not (25 <= lat <= 49):
        raise HTTPException(
            status_code=400,
            detail="Latitude must be 25-49 (USA only)"
        )
    if not (-124 <= lng <= -67):
        raise HTTPException(
            status_code=400,
            detail="Longitude must be -124 to -67 (USA only)"
        )

    data = get_prediction_for_location(lat, lng)
    if not data:
        raise HTTPException(status_code=404, detail="No data found")

    risk_level, color, alert, message = get_risk_info(data['fire_prob'])

    return {
        "requested":  {"latitude": lat, "longitude": lng},
        "matched": {
            "latitude":     data['matched_lat'],
            "longitude":    data['matched_lng'],
            "distance_km":  round(data['distance_deg'] * 111, 1)
        },
        "prediction": {
            "fire_probability":    data['fire_prob'],
            "raw_probability":     data['raw_prob'],
            "fire_percentage":     round(data['fire_prob'] * 100, 1),
            "risk_level":          risk_level,
            "color":               color,
            "alert":               alert,
            "message":             message,
            "data_date":           data['date'],
            "fire_zone":           data['fire_zone'],
            "true_fire_rate":      data['true_fire_rate'],
            "calibrated_threshold": data['calibrated_threshold']
        },
        "history": {
            "peak_fire_prob": data['peak_fire_prob'],
            "peak_date":      data['peak_date'],
            "monthly_avg":    data['monthly_avg'],
            "total_samples":  data['total_samples']
        }
    }


# ══════════════════════════════════════════════════════════════
# ROUTE 11 — Stats
# ══════════════════════════════════════════════════════════════
@app.get("/api/stats", summary="Dataset & Model Statistics", tags=["Data"])
async def get_stats():
    try:
        total      = len(predictions_df)
        fire_preds = int((predictions_df['prediction'] == 1).sum())
        true_fires = int((predictions_df['true_fire']  == 1).sum())
        avg_prob   = float(predictions_df['fire_prob'].mean())
        max_prob   = float(predictions_df['fire_prob'].max())

        yearly = predictions_df.groupby('year').agg(
            fire_predictions = ('prediction', 'sum'),
            true_fires       = ('true_fire',  'sum'),
            avg_prob         = ('fire_prob',  'mean')
        ).reset_index()

        monthly = predictions_df.groupby('month').agg(
            fire_predictions = ('prediction', 'sum'),
            avg_prob         = ('fire_prob',  'mean')
        ).reset_index()

        return {
            "dataset": {
                "total_samples":    total,
                "fire_predictions": fire_preds,
                "true_fires":       true_fires,
                "unique_locations": 36854,
                "date_range":       "2014-2025",
                "avg_fire_prob":    round(avg_prob, 4),
                "max_fire_prob":    round(max_prob, 4),
            },
            "model": {
                "architecture": "ConvLSTM2D",
                "auc":          0.9082,
                "recall":       0.994,
                "precision":    0.150,
                "f1_score":     0.353,
                "threshold":    0.90,
                "epochs":       31,
            },
            "yearly_breakdown":  yearly.to_dict('records'),
            "monthly_breakdown": monthly.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# ROUTE 12 — Trajectory Seeds
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/trajectory",
    summary="Get Fire Trajectory Seeds",
    tags=["Trajectory"]
)
async def get_trajectory(
    year:   int           = Query(2015),
    month:  int           = Query(8, ge=1, le=12),
    region: Optional[str] = Query(None),
    limit:  int           = Query(20, ge=1, le=100)
):
    try:
        filtered = predictions_df[
            (predictions_df['year']      == year)  &
            (predictions_df['month']     == month) &
            (predictions_df['fire_prob'] >  0.85)
        ].copy()

        if region and region in REGION_BOUNDS:
            la, lb, la2, lb2 = REGION_BOUNDS[region]
            filtered = filtered[
                (filtered['latitude']  >= la) &
                (filtered['latitude']  <= lb) &
                (filtered['longitude'] >= la2) &
                (filtered['longitude'] <= lb2)
            ]

        if len(filtered) == 0:
            return {"count": 0, "seeds": []}

        seeds = filtered.nlargest(limit, 'fire_prob')
        return {
            "year":       year,
            "month":      month,
            "region":     region,
            "count":      len(seeds),
            "center_lat": round(seeds['latitude'].mean(),  3),
            "center_lng": round(seeds['longitude'].mean(), 3),
            "seeds": [
                {
                    'latitude':  round(float(r['latitude']),  4),
                    'longitude': round(float(r['longitude']), 4),
                    'fire_prob': round(float(r['fire_prob']), 4),
                    'date':      str(r['date'].date())
                }
                for _, r in seeds.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# ROUTE 13 — Monthly Risk
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/monthly-risk",
    summary="Monthly Fire Risk Breakdown",
    tags=["Data"]
)
async def monthly_risk(year: int = Query(2015)):
    try:
        year_data = predictions_df[predictions_df['year'] == year]
        if len(year_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {year}")

        monthly = year_data.groupby('month').agg(
            avg_prob         = ('fire_prob',  'mean'),
            max_prob         = ('fire_prob',  'max'),
            fire_predictions = ('prediction', 'sum'),
            total_samples    = ('prediction', 'count'),
            true_fires       = ('true_fire',  'sum')
        ).reset_index()

        monthly['fire_rate'] = (
            monthly['fire_predictions'] / monthly['total_samples']
        ).round(4)

        month_names = {
            1:'January', 2:'February', 3:'March', 4:'April',
            5:'May', 6:'June', 7:'July', 8:'August',
            9:'September', 10:'October', 11:'November', 12:'December'
        }
        monthly['month_name'] = monthly['month'].map(month_names)

        return {
            "year":       year,
            "monthly":    monthly.to_dict('records'),
            "peak_month": month_names[
                int(monthly.loc[monthly['avg_prob'].idxmax(), 'month'])
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ADD import at top of app.py:
from gridmet_service import fetch_gridmet_window, test_gridmet_connection

# ── ROUTE: Live Prediction Using GridMET ───────────────────────
@app.post(
    "/api/predict-live",
    summary="Live Fire Prediction Using Real GridMET Weather",
    tags=["Prediction"]
)
async def predict_live(
    lat: float = Query(..., example=37.7749,
                       description="Latitude (USA: 25-49)"),
    lng: float = Query(..., example=-119.4194,
                       description="Longitude (USA: -124 to -67)")
):
    """
    Fetches REAL last 75 days of weather from GridMET API
    for the given location and runs the ConvLSTM2D model.

    This is a TRUE live prediction — no hardcoded values!
    Uses the exact same 17 features as training data.
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )

        # Validate bounds
        if not (25 <= lat <= 49):
            raise HTTPException(
                status_code=400,
                detail="Latitude must be 25-49 (USA only)"
            )
        if not (-124 <= lng <= -67):
            raise HTTPException(
                status_code=400,
                detail="Longitude must be -124 to -67 (USA only)"
            )

        logger.info(
            f"Live prediction request: ({lat}, {lng})"
        )

        # ── Fetch real weather from GridMET ───────────────────
        logger.info("Fetching GridMET weather data...")
        x = fetch_gridmet_window(lat, lng, window_days=75)

        # ── Run real model prediction ──────────────────────────
        logger.info("Running ConvLSTM2D prediction...")
        prob = float(
            model(
                tf.constant(x), training=False
            ).numpy()[0][0]
        )

        risk_level, color, alert, message = get_risk_info(prob)

        logger.info(
            f"✅ Live prediction: "
            f"prob={prob:.4f} "
            f"risk={risk_level}"
        )

        return {
            "latitude":          lat,
            "longitude":         lng,
            "fire_probability":  round(prob, 4),
            "fire_percentage":   round(prob * 100, 1),
            "risk_level":        risk_level,
            "color":             color,
            "alert":             alert,
            "message":           message,
            "data_source":       "GridMET Real Weather API",
            "window_days":       75,
            "note": (
                "Prediction based on real weather data "
                "from the past 75 days at this location. "
                "GridMET data has ~5 day processing lag."
            )
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Live prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── ROUTE: GridMET Health Check ────────────────────────────────
@app.get(
    "/api/gridmet-status",
    summary="Check GridMET API Connection",
    tags=["System"]
)
async def gridmet_status():
    """Check if GridMET API is accessible."""
    is_connected = test_gridmet_connection()
    return {
        "gridmet_connected": is_connected,
        "message": (
            "✅ GridMET API accessible! "
            "Live predictions available."
            if is_connected else
            "❌ GridMET API not accessible. "
            "Using CSV lookup as fallback."
        )
    }


# ══════════════════════════════════════════════════════════════
# ROUTE: Live GridMET Prediction
# Fetches real 75-day weather data and runs model prediction
# ══════════════════════════════════════════════════════════════
@app.get(
    "/api/gridmet-predict",
    summary="Get Live Fire Prediction from GridMET",
    tags=["Prediction"]
)
async def gridmet_predict(
    latitude: float = Query(..., ge=25, le=49, example=37.7749),
    longitude: float = Query(..., ge=-124, le=-67, example=-119.4194),
    location_name: str = Query(default="Custom Location", example="California")
):
    """
    Fetches real 75-day weather data from GridMET API
    and runs ConvLSTM2D model prediction for fire risk.
    
    Returns live fire probability and risk level for the location.
    Falls back to CSV data if GridMET is unavailable.
    """
    try:
        # Validate USA bounds
        if not (25 <= latitude <= 49):
            raise HTTPException(
                status_code=400,
                detail=f"Latitude {latitude} out of USA bounds (25-49)"
            )
        if not (-124 <= longitude <= -67):
            raise HTTPException(
                status_code=400,
                detail=f"Longitude {longitude} out of USA bounds (-124 to -67)"
            )

        logger.info(
            f"🔥 Live GridMET prediction: "
            f"({latitude}, {longitude}) - {location_name}"
        )

        # Fetch 75-day real weather window from GridMET
        try:
            x = fetch_gridmet_window(latitude, longitude)
            data_source = "GridMET Live"
            logger.info(f"✅ GridMET window fetched successfully")
        except Exception as e:
            logger.warning(f"⚠️ GridMET fetch failed, using CSV: {e}")
            # Fallback: use CSV prediction
            pred_data = get_prediction_for_location(latitude, longitude)
            if pred_data is None:
                raise HTTPException(
                    status_code=404,
                    detail="No data for this location"
                )
            prob = pred_data['fire_prob']
            data_source = "CSV Fallback"
            
            risk_level, color, alert, message = get_risk_info(prob)
            return {
                "fire_probability":  round(prob, 4),
                "fire_percentage":   round(prob * 100, 1),
                "risk_level":        risk_level,
                "color":             color,
                "alert":             alert,
                "message":           message,
                "data_source":       data_source,
                "fire_zone":         pred_data['fire_zone'],
                "true_fire_rate":    pred_data['true_fire_rate'],
                "location_name":     location_name,
                "latitude":          latitude,
                "longitude":         longitude,
                "data_date":         pred_data['date'],
                "note":              "Using CSV historical data. GridMET API unavailable."
            }

        # Run model prediction on GridMET window
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prob = float(model(tf.constant(x), training=False).numpy()[0][0])
        
        risk_level, color, alert, message = get_risk_info(prob)

        logger.info(
            f"🔥 Live prediction: {location_name} = "
            f"{round(prob*100,1)}% risk ({risk_level})"
        )

        return {
            "fire_probability":  round(prob, 4),
            "fire_percentage":   round(prob * 100, 1),
            "risk_level":        risk_level,
            "color":             color,
            "alert":             alert,
            "message":           message,
            "data_source":       data_source,
            "location_name":     location_name,
            "latitude":          latitude,
            "longitude":         longitude,
            "timestamp":         str(datetime.now()),
            "note":              "Real weather data from GridMET for the last 75 days"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ GridMET prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
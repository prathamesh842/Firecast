# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///firecast.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# ── Table 1: Subscribers ───────────────────────────────────────
class Subscriber(Base):
    __tablename__ = "subscribers"

    id          = Column(Integer, primary_key=True, index=True)
    email       = Column(String, unique=True, index=True)
    latitude    = Column(Float)
    longitude   = Column(Float)
    location_name = Column(String, default="Custom Location")
    threshold   = Column(Float, default=0.7)
    active      = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

# ── Table 2: Alert History ─────────────────────────────────────
class AlertHistory(Base):
    __tablename__ = "alert_history"

    id           = Column(Integer, primary_key=True, index=True)
    email        = Column(String)
    latitude     = Column(Float)
    longitude    = Column(Float)
    fire_prob    = Column(Float)
    risk_level   = Column(String)
    alert_sent   = Column(Boolean, default=False)
    sent_at      = Column(DateTime, default=datetime.utcnow)

# ── Table 3: Scheduler Logs ────────────────────────────────────
class SchedulerLog(Base):
    __tablename__ = "scheduler_logs"

    id                  = Column(Integer, primary_key=True, index=True)
    run_date            = Column(DateTime, default=datetime.utcnow)
    subscribers_checked = Column(Integer,  default=0)
    alerts_sent         = Column(Integer,  default=0)
    errors              = Column(Integer,  default=0)
    status              = Column(String,   default="running")
    finished_at         = Column(DateTime, nullable=True)

# ── Create all tables ──────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized!")
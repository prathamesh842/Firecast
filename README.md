# 🔥 FireCast - Wildfire Risk Prediction System

A real-time wildfire risk prediction and alert system built with FastAPI, TensorFlow, and GridMet weather data integration. FireCast uses machine learning to forecast fire risk across different regions and sends email alerts to subscribers.

## 🌟 Features

- **ML-Powered Risk Prediction**: TensorFlow-based model predicting wildfire probability at specific locations
- **Real-Time Weather Integration**: Fetches GridMet weather data for accurate predictions
- **Email Alert System**: Automated email notifications for subscribers when fire risk exceeds threshold
- **Scheduled Monitoring**: APScheduler-based automated daily/periodic checks
- **Interactive Maps**: Folium-based trajectory and risk visualization maps
- **REST API**: FastAPI endpoints for predictions, subscriptions, and alert management
- **Database Persistence**: SQLite database for subscriber management and alert history

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **ML**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Scheduling**: APScheduler
- **Database**: SQLite + SQLAlchemy ORM
- **Web Framework**: HTML/Static assets
- **Visualization**: Folium (for interactive maps)

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.18.0
- FastAPI 0.115.0
- See `requirements.txt` for full dependencies

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd final_fire
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Database

```bash
python -c "from database import init_db; init_db()"
```

### 4. Run the Server

```bash
uvicorn app:app --reload --port 8000
```

Server will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## 📁 Project Structure

```
final_fire/
├── app.py                 # FastAPI main application & scheduler
├── database.py            # SQLAlchemy models (Subscriber, AlertHistory, SchedulerLog)
├── email_service.py       # Email alert functionality
├── gridmet_service.py     # GridMet API integration for weather data
├── requirements.txt       # Python dependencies
├── model/
│   └── best_model_5.0x.h5    # Trained TensorFlow model
├── data/
│   ├── all_predictions.csv    # Cached predictions
│   ├── location_risk.csv      # Location-based risk data
│   └── monthly_trajectory.csv # Historical fire trajectories
├── static/
│   ├── index.html             # Web interface
│   └── trajectory_maps/       # Interactive Folium visualizations
└── Wildfire_Dataset.csv       # Training data
```

## 🔧 Configuration

### Region Bounds
Modify `REGION_BOUNDS` in `app.py` to define geographical regions for predictions:
```python
REGION_BOUNDS = {
    'california': {'north': 42, 'south': 32, 'east': -114, 'west': -125},
    # Add more regions...
}
```

### Email Configuration
Configure email settings in `email_service.py`:
- SMTP server
- Sender credentials
- Alert thresholds

### Model Path
Update `MODEL_PATH` in `app.py` to point to your trained model.

## 📊 API Endpoints

### Predictions
- `GET /predictions` - Get all cached predictions
- `GET /predict/{latitude}/{longitude}` - Get risk for specific location
- `GET /region/{region_name}` - Get predictions for entire region

### Subscribers
- `POST /subscribe` - Register for alerts
- `GET /subscribers` - List all subscribers
- `DELETE /subscriber/{subscriber_id}` - Unsubscribe

### Admin
- `GET /scheduler-logs` - View scheduler execution logs
- `GET /test-email` - Send test email

## 🔄 Scheduler Tasks

The application runs automated tasks via APScheduler:

- **Daily Checks**: Verifies fire risk for all subscribers
- **Alert Sending**: Sends emails when risk exceeds threshold
- **Log Tracking**: Records scheduler execution and statistics

Check logs at `/scheduler-logs` endpoint.

## 📧 Email Alerts

Subscribers receive alerts containing:
- Current fire risk probability
- Risk level (Low/Medium/High/Critical)
- Location coordinates
- Recommended actions

Threshold can be customized per subscriber (default: 0.7).

## 🗺️ Visualization

Pre-generated interactive maps available in `/static/trajectory_maps/`:
- Fire trajectory predictions
- Historical fire paths
- Risk heatmaps by season
- Regional fire patterns

## 🧪 Testing

Run test scripts to verify functionality:

```bash
python test_gridmet.py      # Test GridMet API integration
python test_model.py        # Test model predictions
python fire_testing.py      # General fire prediction tests
python csv_checking.py      # Validate CSV data
```

## 📝 Database Schema

### Subscribers Table
- id, email (unique), latitude, longitude
- location_name, threshold, active status
- created_at timestamp

### Alert History Table
- id, email, coordinates, fire_probability
- risk_level, alert_sent status, sent_at timestamp

### Scheduler Logs Table
- id, run_date, subscribers_checked
- alerts_sent, errors, status, finished_at

## 🐛 Troubleshooting

**Model not found**: Ensure `best_model_5.0x.h5` exists in `/model` directory

**GridMet connection issues**: Check internet connection and GridMet API availability

**Email not sending**: Verify SMTP credentials and email service configuration

**Database locked**: Delete `firecast.db` and reinitialize

## 📄 License

MIT License - feel free to use and modify

## 👤 Author

FireCast Development Team

## 🤝 Contributing

Contributions welcome! Submit issues and pull requests.

---

**Last Updated**: April 2026

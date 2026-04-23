# email_service.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging

logger = logging.getLogger(__name__)

# ── YOUR GMAIL CREDENTIALS ─────────────────────────────────────
GMAIL_ADDRESS  = "prathameshbaviskar817@gmail.com"      # ← your Gmail
GMAIL_APP_PASS = "emcl vcwk yqia qqob" # ← your App Password

def send_fire_alert_email(
    to_email,
    location_name,
    latitude,
    longitude,
    fire_prob,
    risk_level,
    temperature=None,
    humidity=None,
    wind_speed=None,
    extra_info=None
):
    """
    Sends a professional fire alert email.
    """
    try:
        pct   = round(fire_prob * 100, 1)
        color = (
            '#ff2200' if risk_level == 'EXTREME' else
            '#ff6600' if risk_level == 'HIGH'    else
            '#ffaa00' if risk_level == 'MODERATE' else
            '#00ff88'
        )
        emoji = (
            '🔴' if risk_level == 'EXTREME' else
            '🟠' if risk_level == 'HIGH'    else
            '🟡' if risk_level == 'MODERATE' else
            '🟢'
        )

        # ── HTML Email Template ────────────────────────────────
        html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width">
</head>
<body style="margin:0;padding:0;background:#0a0a0a;
             font-family:Arial,sans-serif;">

  <!-- Header -->
  <div style="background:#111;padding:32px;text-align:center;
              border-bottom:3px solid {color};">
    <h1 style="color:{color};font-size:36px;
               letter-spacing:4px;margin:0;">
      🔥 FIRECAST
    </h1>
    <p style="color:#666;font-size:12px;
              letter-spacing:2px;margin:8px 0 0;">
      WILDFIRE PREDICTION SYSTEM
    </p>
  </div>

  <!-- Alert Banner -->
  <div style="background:{color}22;padding:24px 32px;
              border-left:4px solid {color};margin:24px;">
    <h2 style="color:{color};margin:0;font-size:22px;">
      {emoji} {risk_level} FIRE RISK DETECTED
    </h2>
    <p style="color:#aaa;margin:8px 0 0;font-size:14px;">
      Our ConvLSTM2D model has detected elevated
      wildfire risk in your subscribed area.
    </p>
  </div>

  <!-- Details Card -->
  <div style="background:#141414;margin:0 24px;
              padding:28px;border:1px solid #2a2a2a;">

    <table style="width:100%;border-collapse:collapse;">
      <tr>
        <td style="padding:12px 0;border-bottom:1px solid #222;
                   color:#666;font-size:12px;
                   letter-spacing:1px;text-transform:uppercase;">
          Location
        </td>
        <td style="padding:12px 0;border-bottom:1px solid #222;
                   color:#f0ede8;font-size:14px;text-align:right;">
          {location_name}<br>
          <span style="color:#555;font-size:12px;">
            {latitude:.4f}°N, {longitude:.4f}°W
          </span>
        </td>
      </tr>
      <tr>
        <td style="padding:12px 0;border-bottom:1px solid #222;
                   color:#666;font-size:12px;
                   letter-spacing:1px;text-transform:uppercase;">
          Fire Probability
        </td>
        <td style="padding:12px 0;border-bottom:1px solid #222;
                   color:{color};font-size:28px;
                   font-weight:bold;text-align:right;">
          {pct}%
        </td>
      </tr>
      <tr>
        <td style="padding:12px 0;border-bottom:1px solid #222;
                   color:#666;font-size:12px;
                   letter-spacing:1px;text-transform:uppercase;">
          Risk Level
        </td>
        <td style="padding:12px 0;border-bottom:1px solid #222;
                   text-align:right;">
          <span style="background:{color}22;color:{color};
                       padding:4px 16px;font-size:13px;
                       font-weight:bold;letter-spacing:2px;">
            {risk_level}
          </span>
        </td>
      </tr>
      {'<tr><td style="padding:12px 0;border-bottom:1px solid #222;color:#666;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Temperature</td><td style="padding:12px 0;border-bottom:1px solid #222;color:#f0ede8;font-size:14px;text-align:right;">' + str(temperature) + '°C</td></tr>' if temperature else ''}
      {'<tr><td style="padding:12px 0;border-bottom:1px solid #222;color:#666;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Humidity</td><td style="padding:12px 0;border-bottom:1px solid #222;color:#f0ede8;font-size:14px;text-align:right;">' + str(humidity) + '%</td></tr>' if humidity else ''}
      {'<tr><td style="padding:12px 0;border-bottom:1px solid #222;color:#666;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Wind Speed</td><td style="padding:12px 0;border-bottom:1px solid #222;color:#f0ede8;font-size:14px;text-align:right;">' + str(wind_speed) + ' m/s</td></tr>' if wind_speed else ''}
      {'<tr><td style="padding:12px 0;border-bottom:1px solid #222;color:#666;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Data Date</td><td style="padding:12px 0;border-bottom:1px solid #222;color:#f0ede8;font-size:14px;text-align:right;">' + str(extra_info.get("data_date","N/A")) + '</td></tr>' if extra_info else ''}
      {'<tr><td style="padding:12px 0;color:#666;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Dry Days</td><td style="padding:12px 0;color:#f0ede8;font-size:14px;text-align:right;">' + str(extra_info.get("dry_days","N/A")) + ' days</td></tr>' if extra_info else ''}
    </table>
  </div>

  <!-- Model Info -->
  <div style="margin:16px 24px;padding:16px;
              background:#111;border:1px dashed #2a2a2a;">
    <p style="color:#555;font-size:11px;margin:0;
              text-align:center;letter-spacing:1px;">
      🤖 Predicted by ConvLSTM2D Model &nbsp;|&nbsp;
      AUC: 0.9082 &nbsp;|&nbsp; Recall: 99.4%
    </p>
  </div>

  <!-- Footer -->
  <div style="padding:24px 32px;text-align:center;
              border-top:1px solid #1e1e1e;margin-top:24px;">
    <p style="color:#333;font-size:11px;margin:0;">
      FireCast Wildfire Prediction System<br>
      You are receiving this because you subscribed
      to fire alerts for this location.
    </p>
  </div>

</body>
</html>
        """

        # ── Plain text fallback ────────────────────────────────
        text = f"""
FIRECAST — WILDFIRE ALERT

{risk_level} FIRE RISK DETECTED

Location:    {location_name}
Coordinates: {latitude:.4f}N, {longitude:.4f}W
Probability: {pct}%
Risk Level:  {risk_level}

Predicted by ConvLSTM2D Model (AUC: 0.9082)
        """

        # ── Build email ────────────────────────────────────────
        msg = MIMEMultipart('alternative')
        msg['Subject'] = (
            f"🔥 FireCast Alert — {risk_level} Fire Risk "
            f"at {location_name} ({pct}%)"
        )
        msg['From']    = f"FireCast Alerts <{GMAIL_ADDRESS}>"
        msg['To']      = to_email

        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))

        # ── Send via Gmail SMTP ────────────────────────────────
        with smtplib.SMTP_SSL(
            'smtp.gmail.com', 465
        ) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(
                GMAIL_ADDRESS,
                to_email,
                msg.as_string()
            )

        logger.info(f"✅ Alert email sent to {to_email}")
        return True

    except Exception as e:
        logger.error(f"❌ Email failed: {e}")
        return False


def send_test_email(to_email):
    """
    Sends a simple test email to verify setup.
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "✅ FireCast — Email Alert System Working!"
        msg['From']    = f"FireCast <{GMAIL_ADDRESS}>"
        msg['To']      = to_email

        html = """
<body style="background:#0a0a0a;font-family:Arial;padding:40px;">
  <div style="max-width:500px;margin:0 auto;
              background:#141414;padding:32px;
              border:1px solid #2a2a2a;
              border-top:3px solid #ff6600;">
    <h2 style="color:#ff6600;letter-spacing:3px;">
      🔥 FIRECAST
    </h2>
    <h3 style="color:#f0ede8;">
      ✅ Email Alert System Connected!
    </h3>
    <p style="color:#999;line-height:1.8;">
      Your email is successfully connected to
      the FireCast alert system.<br><br>
      You will receive fire risk alerts when
      wildfire probability crosses your threshold.
    </p>
    <div style="background:#111;padding:16px;
                border-left:3px solid #ff6600;
                margin-top:20px;">
      <p style="color:#555;font-size:12px;margin:0;">
        🤖 ConvLSTM2D Model · AUC 0.9082 · Recall 99.4%
      </p>
    </div>
  </div>
</body>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(
                GMAIL_ADDRESS,
                to_email,
                msg.as_string()
            )

        logger.info(f"✅ Test email sent to {to_email}")
        return True

    except Exception as e:
        logger.error(f"❌ Test email failed: {e}")
        return False
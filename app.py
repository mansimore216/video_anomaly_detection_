import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import smtplib
from email.message import EmailMessage
from flask import Flask, request, jsonify, render_template
from torch.nn.functional import mse_loss
from PIL import Image
from io import BytesIO
import base64

# ================= CONFIG =================
SEQ_LEN = 10
THRESHOLD = 0.015
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = "uploads"
ANOMALY_FOLDER = "static/anomalies"

# -------- EMAIL CONFIG --------
SENDER_EMAIL = "mansimore216@gmail.com"
SENDER_PASSWORD = "fgqs tdpc vbtw preq"  # Gmail App Password
RECEIVER_EMAIL = "mansimore216@gmail.com"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANOMALY_FOLDER, exist_ok=True)

# ================= MODEL =================
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        out = self.decoder(self.encoder(x))
        return out.view(b, s, c, h, w)

model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(torch.load("model_weights.pth", map_location=DEVICE))
model.eval()

# ================= EMAIL FUNCTION =================
def send_email_alert(confidence, frame_name):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Video Anomaly Detected"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(f"""
    An anomaly has been detected in the uploaded video.

    Type : Accident / Abnormal Motion
    Confidence : {confidence}
    Frame : {frame_name}

    Please check the dashboard for details.
    """)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print("✅ Email alert sent")
    except Exception as e:
        print("❌ Email failed:", e)

# ================= UTILS =================
def pil_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

# ================= FLASK =================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # --- READ VIDEO FRAMES (COLOR) ---
    cap = cv2.VideoCapture(video_path)
    frames_color = []
    frames_gray = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_color.append(frame.copy())  # keep color for frontend
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        frames_gray.append(gray)

    cap.release()

    scores = []
    frame_indices = []
    alerts = []
    saved_frames = []
    email_sent = False

    # --- PROCESS SEQ_LEN FRAMES ---
    for i in range(len(frames_gray) - SEQ_LEN):
        clip = frames_gray[i:i + SEQ_LEN]
        clip = np.array(clip) / 255.0
        clip = torch.tensor(clip).unsqueeze(0).unsqueeze(2).float().to(DEVICE)  # (1,S,1,H,W)

        with torch.no_grad():
            recon = model(clip)
            loss = mse_loss(recon, clip).item()

        scores.append(round(loss, 4))
        frame_indices.append(i + SEQ_LEN // 2)  # middle frame index for graph

        # Detect anomaly
        if loss > THRESHOLD:
            idx = i + SEQ_LEN // 2
            frame_name = f"anomaly_{idx}.jpg"
            frame_path = os.path.join(ANOMALY_FOLDER, frame_name)
            cv2.imwrite(frame_path, frames_color[idx])  # save color frame

            confidence = round(min(loss * 50, 1.0), 2)

            alerts.append({
                "frame": frame_name,
                "confidence": confidence,
                "type": "Accident / Abnormal Motion"
            })

            saved_frames.append(frame_path.replace("\\", "/"))

            if not email_sent:
                send_email_alert(confidence, frame_name)
                email_sent = True

    # --- SELECT 3 KEY FRAMES (START, PEAK, END) ---
    if saved_frames:
        start_idx = 0
        peak_idx = np.argmax(scores)
        end_idx = len(saved_frames) - 1

        frames_base64 = [
            pil_to_base64(Image.fromarray(cv2.cvtColor(frames_color[start_idx], cv2.COLOR_BGR2RGB))),
            pil_to_base64(Image.fromarray(cv2.cvtColor(frames_color[peak_idx], cv2.COLOR_BGR2RGB))),
            pil_to_base64(Image.fromarray(cv2.cvtColor(frames_color[end_idx], cv2.COLOR_BGR2RGB))),
        ]
    else:
        frames_base64 = []

    return jsonify({
        "scores": scores,
        "frame_indices": frame_indices,
        "alerts": alerts,
        "frames": frames_base64
    })

if __name__ == "__main__":
    app.run(debug=True)
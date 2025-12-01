# ---------------------------------------------------------
# FAST & RELIABLE REALTIME PIPELINE (NO LAG)
# Face Detection = MTCNN → DeepFace → Haar (fallback)
# Emotion = your CNN model (optimized with frame skipping)
# Age + Gender = DeepFace (stable)
# ---------------------------------------------------------

import cv2
import torch
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from collections import deque

from model import MultiTaskCNN
from datasets import val_transform, FIXED_EMOTION_ORDER

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True   # SPEED BOOST


# ---------------------------
# Load Emotion Model
# ---------------------------
EMOTIONS = FIXED_EMOTION_ORDER
model = MultiTaskCNN(num_emotions=len(EMOTIONS))
model.load_state_dict(torch.load("models/multitask_cnn.pth", map_location=device))
model.to(device)
model.eval()

# ---------------------------
# Buffers
# ---------------------------
emotion_buf = deque(maxlen=8)
age_buf = deque(maxlen=10)
gender_buf = deque(maxlen=10)

def mode(buf):
    return max(set(buf), key=buf.count) if buf else "?"

def avg(buf):
    return int(sum(buf) / len(buf)) if buf else "?"


# ---------------------------
# Frame-Skipping for FAST EMOTION
# ---------------------------
FRAME_INTERVAL = 3
frame_index = 0


# ---------------------------
# Face Detectors
# ---------------------------
mtcnn = MTCNN()
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------------------------
# Camera
# ---------------------------
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")


# ---------------------------
# LOOP
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_index += 1
    faces = []

    # --------- TRY MTCNN ---------
    try:
        mt = mtcnn.detect_faces(frame)
        for f in mt:
            (x, y, w, h) = f["box"]
            faces.append((x, y, w, h))
    except:
        pass

    # --------- TRY DeepFace detector ---------
    if len(faces) == 0:
        try:
            df_face = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)
            if df_face:
                r = df_face[0]["facial_area"]
                faces.append((r["x"], r["y"], r["w"], r["h"]))
        except:
            pass

    # --------- TRY Haar ---------
    if len(faces) == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = haar.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in haar_faces:
            faces.append((x, y, w, h))

    # No face found
    if len(faces) == 0:
        cv2.imshow("Real-Time Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue


    # ---------------------------
    # AGE + GENDER (DeepFace)
    # ---------------------------
    try:
        df = DeepFace.analyze(
            img_path = frame,
            actions=['age', 'gender'],
            enforce_detection=False,
            detector_backend='opencv'
        )[0]

        age_buf.append(df["age"])
        gender_buf.append(df["dominant_gender"])

        smooth_age = avg(age_buf)
        smooth_gender = mode(gender_buf)

    except Exception as e:
        print("DeepFace failed:", e)
        smooth_age = "?"
        smooth_gender = "?"


    # ---------------------------
    # EMOTION (Your CNN) — every Nth frame
    # ---------------------------
    for (x, y, w, h) in faces:

        # ROI
        roi = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Only run emotion every FRAME_INTERVAL
        if frame_index % FRAME_INTERVAL == 0:

            img_t = val_transform(roi_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)

            idx = int(torch.argmax(out["emotion"]).cpu().numpy())
            emotion_buf.append(EMOTIONS[idx])

        smooth_emotion = mode(emotion_buf)

        # ---------------- Draw ----------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame, smooth_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.putText(frame, f"Age: {smooth_age}", (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, smooth_gender, (x, y+h+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Real-Time Emotion + Age + Gender", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

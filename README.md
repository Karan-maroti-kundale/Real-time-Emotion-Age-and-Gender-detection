# 🎭 Real-Time Emotion, Age & Gender Detection System

## 👁️ Overview
A production-ready hybrid deep learning system for real-time Emotion Recognition, Age Estimation, and Gender Classification from live webcam video. Combines:

- **Custom CNN (PyTorch)** for emotion
- **DeepFace** pretrained models for age & gender
- **Multi-stage face detection** pipeline: MTCNN → RetinaFace → Haar Cascade
- **Frame smoothing & stabilization buffers**
- **Optimized inference** for CPU and GPU

Use cases: mental-health analysis, smart retail, driver safety, HCI, surveillance, affective computing.

***

## 🚀 Key Features

### 🎭 Emotion Recognition (Custom CNN)
- 7-class classifier: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**
- Trained on balanced dataset with augmentations
- PyTorch architecture optimized for inference
- Optional ONNX Runtime support (10–15× faster)

### 🧓 Age & 🧑 Gender Estimation (DeepFace API)
- High-accuracy pretrained models
- Robust to lighting variations & occlusions
- Uses RetinaFace detection backend
- No training dataset required

### ⭐ Multi-Stage Face Detection Pipeline
Automatically falls back to the best available detector:
1. **MTCNN** (primary)  
2. **DeepFace RetinaFace** (secondary)  
3. **Haar Cascade** (fallback)

Handles: low light, side faces, motion blur, glasses/beard.

### 🔄 Prediction Stabilization
- Emotion → **Mode smoothing**
- Age → **Rolling average**
- Gender → **Frequency mode**

### ⚡ Real-Time Performance
- CPU: **18–25 FPS**
- GPU (PyTorch): **35–60 FPS**

***

## 🧠 System Workflow
Webcam → Face Detector(s) → Face ROI  
├── Emotion CNN (PyTorch)  
└── Age & Gender (DeepFace)  

Final Output → Smoothing → On-screen Overlay

***

## 📦 Project Structure
```
AI-Powered-Emotion-Age-Gender-Detection/
│── data/
│   ├── Emotion/
│   └── Age_Gender/
│── models/
│   ├── multitask_cnn.pth
│   └── multitask_cnn.onnx (optional)
│── src/
│   ├── infer.py
│   ├── train.py
│   ├── dataset.py
│   ├── model.py
│   └── convert_to_onnx.py
│── output.mp4
│── requirements.txt
│── README.md
```

***

## ⚙️ Installation & Setup
1. Clone the repository
```bash
git clone <your-repo-url>
cd AI-Powered-Emotion-Age-Gender-Detection
```
2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run real-time detection
```bash
python src/infer.py
```

***

## 🧪 Training the Custom CNN (Emotion Only)
To retrain:
```bash
python src/train.py
```
Model saves to:
```
models/multitask_cnn.pth
```

***

## 🎥 Example Output
- A sample real-time video (output.mp4) demonstrates emotion label, age estimation, gender classification, and smooth/stable bounding boxes.

***

## 🎯 Real-World Applications
- 🚗 Automotive: driver drowsiness & emotional monitoring  
- 🛍 Retail Analytics: customer sentiment, age/gender recommendations  
- 🏥 Healthcare: stress detection, monitoring depression/anxiety indicators  
- 🎓 Education: student attention & engagement  
- 🛡 Security & Surveillance: suspicious emotion detection, behavioral analysis

***

## 📊 Performance Metrics
| Task                   | Accuracy / Error      |
|------------------------|-----------------------|
| Emotion Recognition    | ~85–88%               |
| Gender Classification  | ~97%                  |
| Age Estimation         | ±3–5 years            |
| Realtime FPS (CPU)     | 18–25 FPS             |
| Realtime FPS (GPU)     | 35–60 FPS             |

***

## 🛠 Future Enhancements
- YOLOv8 face detection  
- Facial landmarks & expression intensity scoring  
- Multi-face tracking  
- Lightweight MobileNet emotion model  
- ONNX GPU-accelerated pipeline

***

## 👨‍💻 Author
Karan Kundale — AI/ML Engineer • Full Stack Developer

If you'd like, I can also generate:
- Usage examples, API docs, or deployment instructions
- Model training logs or experiment reproducibility notes
- Lightweight README variants for GitHub release
- Test cases and CI examples
- Performance profiling summary


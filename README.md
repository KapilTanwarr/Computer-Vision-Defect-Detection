# 🚀 Computer Vision-Based Product Defect Detection System

A deep learning-powered quality inspection tool that classifies product images as **OK** or **Defect** in real time, boosting inspection efficiency and accuracy.

## 📌 Highlights
- CNN model (TensorFlow/Keras) with data augmentation
- OpenCV preprocessing + optional Grad-CAM visual explanations
- Streamlit web app for drag‑and‑drop prediction
- Trained model saved as `model/defect_cnn.h5`

## 🏗️ Project Layout
```
cv-defect-detection/
├── data/          # place train/val images here
│   ├── train/
│   │   ├── OK/
│   │   └── Defect/
│   └── val/
│       ├── OK/
│       └── Defect/
├── model/
│   └── defect_cnn.h5   # saved after training
├── app/
│   └── streamlit_app.py
├── defect_detector.py
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
python defect_detector.py --mode train --data_dir data --epochs 10
```

### 3. Run App
```bash
streamlit run app/streamlit_app.py
```

Upload a product image and view prediction + Grad-CAM.

## 🧠 Tech Stack
TensorFlow · Keras · OpenCV · Streamlit · Grad‑CAM · Python



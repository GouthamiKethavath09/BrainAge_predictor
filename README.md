# 🧠 Brain Health Analyzer (MRI-based AI System)

## 📌 Overview
The Brain Health Analyzer is an AI-powered web application that predicts a person's brain age using MRI scan data (.nii / .nii.gz files). It compares predicted brain age with actual age to assess brain health and detect potential neurological risks.

This system uses Deep Learning (3D CNN) and provides:
- Brain age prediction
- Brain Age Gap (BAG)
- AI-based disease analysis
- MRI visualization
- PDF report generation

---

## 🚀 Features

### 🧠 MRI Upload
- Supports `.nii` and `.nii.gz` formats
- Upload real brain MRI scans

### 🔍 MRI Visualization
- Slice-based viewing using slider
- Multi-slice brain visualization
- Helps in understanding scan structure

### 📊 Brain Age Prediction
- Uses 3D Convolutional Neural Network
- Predicts brain age from MRI volume

### 📈 Brain Age Gap (BAG)
- BAG = Predicted Age - Actual Age  
- Interpretation:
  - Positive → Faster aging
  - Negative → Younger brain

### 🤖 AI-Based Disease Detection
Smart classification based on BAG:
- Alzheimer’s Risk
- Early Dementia Risk
- Mild Cognitive Impairment
- Normal Aging
- Excellent Brain Health

### 🔥 Brain Activation Map
- Gradient-based visualization
- Highlights important brain regions

### 📄 PDF Report Generation
- Download complete analysis report
- Includes prediction, diagnosis, cause, precautions

### 🎨 UI Features
- Dark theme
- Glassmorphism cards
- Smooth animations
- Clean dashboard layout

---

## 🧪 Tech Stack

- Frontend: Streamlit  
- Backend: Python  
- Deep Learning: PyTorch  
- MRI Handling: Nibabel  
- Visualization: Matplotlib  
- PDF: ReportLab  

---

## 📂 Project Structure

BrainAge_prediction/
│
├── app.py
├── brain_age_model.pth
├── brain.jpg
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/brain-health-analyzer.git
cd brain-health-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
torch
numpy
nibabel
matplotlib
reportlab
```

---

## 🧠 Model Architecture

- 3D Convolutional Neural Network  
- Conv3D + ReLU + MaxPooling  
- Adaptive Average Pooling  
- Fully Connected Layers  

Input: 3D MRI volume  
Output: Predicted Brain Age  

---

## 📊 How It Works

1. Upload MRI scan  
2. Normalize + resize image  
3. Model predicts brain age  
4. Calculate Brain Age Gap  
5. AI detects condition  
6. Show results + visualization  
7. Generate PDF report  

---

## ⚠️ Disclaimer

- For educational and research purposes only  
- Not for medical diagnosis  
- Always consult a doctor  

---

## 🌟 Future Improvements

- Real clinical dataset  
- Multi-disease classification  
- 3D brain visualization  
- Cloud GPU deployment  
- User login system  

---


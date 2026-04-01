# 🧠 Brain Health Analyzer (MRI-based AI System)

## 📌 Overview
The Brain Health Analyzer is an AI-powered web application that predicts a person's brain age using MRI scan data (.nii / .nii.gz files). It compares predicted brain age with actual age to assess brain health and detect potential neurological risks.
### live link: https://brainagepredictor-kmbwyisz9xcf9hibtnvjz5.streamlit.app/
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

### 📄  Report Generation
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


## 🌟 Future Improvements

- Real clinical dataset  
- Multi-disease classification  
- 3D brain visualization  
- Cloud GPU deployment  
- User login system  

---


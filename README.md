# 🍅 Tomato Leaf Disease Detection System

> **BSc Dissertation Project** · University of Zimbabwe · 2025
> Deployed as a **Streamlit web application** for real-time inference

A deep learning–based image classification system for detecting tomato leaf diseases from field photographs — built as a BSc final year dissertation and deployed as an accessible web app for practical use in agricultural settings across Zimbabwe and Africa.

---

## 🎯 Problem Statement

Tomato farming is a critical income source for smallholder farmers across Zimbabwe and sub-Saharan Africa. Crop diseases cause significant yield losses, yet farmers in rural areas often lack access to agronomists or diagnostic tools. This system provides an automated solution for early disease detection directly from a photograph, accessible via a simple web interface — no specialist knowledge required.

---

## 🏗️ System Architecture

```
Input Image (field photo)
        ↓
 Preprocessing & Augmentation
        ↓
  CNN Model (TensorFlow/Keras)
        ↓
  Disease Classification
        ↓
 Streamlit Web Application
        ↓
 Real-time Prediction + Confidence Score
```

---

## 🦠 Disease Classes Detected

| Class | Disease |
|-------|---------|
| Bacterial Spot | Xanthomonas bacterial infection |
| Early Blight | Alternaria solani fungal disease |
| Late Blight | Phytophthora infestans |
| Leaf Mold | Passalora fulva |
| Septoria Leaf Spot | Septoria lycopersici |
| Spider Mites | Two-spotted spider mite damage |
| Target Spot | Corynespora cassiicola |
| Yellow Leaf Curl Virus | Tomato yellow leaf curl virus |
| Mosaic Virus | Tomato mosaic virus |
| Healthy | No disease detected |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow / Keras |
| Model Architecture | CNN with transfer learning (ImageNet) |
| Data Augmentation | Keras ImageDataGenerator |
| Deployment | Streamlit web application |
| Language | Python 3.x |

---

## 📊 Model Training

- **Dataset:** PlantVillage dataset (labeled tomato leaf images)
- **Approach:** Transfer learning — pre-trained ImageNet base + custom classification head
- **Augmentation:** Random flips, rotations, zoom, brightness variation
- **Evaluation:** Accuracy, Precision, Recall, F1-score per class
- **Inference:** Real-time single image and batch prediction

---

## 🚀 Running the Application

### Prerequisites
```bash
pip install tensorflow streamlit numpy pillow
```

### Run
```bash
streamlit run app.py
```

Upload a tomato leaf image and the model returns the predicted disease class with a confidence score.

---

## 📁 Repository Structure

```
Tomato-Leaf-Disease-Detector/
├── app.py                    # Streamlit web application
├── model/
│   └── tomato_model.h5       # Trained model weights
├── notebooks/
│   └── training.ipynb        # Model training notebook
├── requirements.txt
└── README.md
```

---

## 🏆 Recognition

- 🎓 **BSc Dissertation Project** — Data Science & Informatics, University of Zimbabwe (2025)
- 🌐 Deployed as a live Streamlit web application

---

## 👤 Author

**Peter Tinashe Mundowa**
Data Scientist & AI Engineer · Harare, Zimbabwe 🇿🇼

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/peter-tinashe-mundowa-758a2b239/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-f5a623?style=flat&logo=github)](https://peterthedatascientist.github.io)
[![Zindi](https://img.shields.io/badge/Zindi-PeterTheAnalyst-1DA462?style=flat)](https://zindi.africa/users/PeterTheAnalyst)

---

*Applying AI to protect food security across Africa.* 🌍

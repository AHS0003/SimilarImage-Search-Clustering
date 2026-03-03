# 🔍 Similar Image Search & Clustering (CBIR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-VGG19-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge&logo=opencv&logoColor=white)

> **Machine Learning & Computer Vision Project**  
> A complete web application for image analysis, clustering, and similar image search, comparing classical descriptors (Color, Shape, Texture) with Deep Learning (VGG19).

---

## 📝 Project Description

This project implements an image processing pipeline capable of structuring a visual database and retrieving images similar to a query (CBIR - Content-Based Image Retrieval).

The main objective is to compare the effectiveness of traditional feature extraction methods with convolutional neural networks (CNNs) through an interactive web interface.

---

## ✨ Key Features

### 🔎 Feature Extraction
- 🎨 **Color:** RGB Histogram (color distribution).
- 📐 **Shape:** Zernike Moments (rotation invariant).
- 🧱 **Texture:** SFTA (Segmentation-based Fractal Texture Analysis).
- 🧠 **Deep Learning:** Features extracted using **VGG19** (Transfer Learning).

### 📊 Unsupervised Clustering
- **K-Means** algorithm for grouping images.
- Optimization of the number of clusters (*k*) using the **Elbow Method**.
- Cluster quality evaluation using the **Silhouette Score**.

### 🔍 Similarity Search
- **KNN (K-Nearest Neighbors)** algorithm to identify the closest images to a query.

### 🌐 Interactive Web Interface
- Complete dashboard to configure analysis.
- Visualization of performance curves.
- Interactive exploration of clustering and similarity results.

---

## 🛠️ Tech Stack

This project was developed using **Python** and the following libraries:

| Domain | Technologies Used |
| :--- | :--- |
| **Web Backend** | `Flask` (REST API) |
| **Frontend** | `HTML5`, `CSS3`, `JavaScript`, `Chart.js` (Visualization) |
| **Image Processing** | `OpenCV` (cv2), `Mahotas` |
| **Machine Learning** | `Scikit-learn` (K-Means, KNN) |
| **Deep Learning** | `TensorFlow` / `Keras` (VGG19 Model) |
| **Computation & Data** | `NumPy`, `Matplotlib` |

---

## 📂 Dataset

The system was trained and tested on a dataset organized into **12 distinct classes**, totaling approximately 1200 images:

- 🚲 Bike  
- 🚌 Bus  
- 🐱 Cat  
- 🦖 Dinosaurs  
- 🐶 Dog  
- 🐘 Elephants  
- 🌸 Flowers  
- 🍔 Foods  
- 🐎 Horses  
- 🏛️ Monuments  
- 🏔️ Mountains & Snow  
- ⌚ Watch  

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/PROJECT_NAME.git
cd PROJECT_NAME
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application
```bash
python app.py
```

### 4️⃣ Open in your browser
```
http://127.0.0.1:5000/
```

---

## 🎯 Project Goals

- Compare classical image descriptors vs Deep Learning features
- Evaluate clustering performance using quantitative metrics
- Implement a full CBIR system
- Develop an interactive ML-powered web dashboard

---

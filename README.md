# 🔍 Recherche d'Images Similaires & Clustering (CBIR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-VGG19-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge&logo=opencv&logoColor=white)

> **Projet de Machine Learning & Vision par Ordinateur**
> Une application web complète pour l'analyse, le clustering et la recherche d'images similaires, comparant des descripteurs classiques (Couleur, Forme, Texture) et le Deep Learning (VGG19).

---

## 📝 Description du Projet

Ce projet implémente un pipeline de traitement d'images capable de structurer une base de données visuelle et de retrouver des images similaires à une requête (CBIR - Content-Based Image Retrieval).

L'objectif principal est de comparer l'efficacité des méthodes d'extraction de caractéristiques traditionnelles face aux réseaux de neurones convolutionnels (CNN) via une interface web interactive.

### ✨ Fonctionnalités Clés

* **Extraction de caractéristiques (Feature Extraction) :**
    * 🎨 **Couleur :** Histogramme RGB (Distribution chromatique).
    * 📐 **Forme :** Moments de Zernike (Invariants par rotation).
    * 🧱 **Texture :** SFTA (Segmentation-based Fractal Texture Analysis).
    * 🧠 **Deep Learning :** Features extraites via **VGG19** (Transfer Learning).
* **Clustering Non-Supervisé :**
    * Utilisation de l'algorithme **K-Means** pour grouper les images.
    * Optimisation du nombre de clusters ($k$) via la **méthode du Coude (Elbow Method)**.
    * Évaluation de la qualité des clusters via le **Score de Silhouette**.
* **Recherche de Similarité :**
    * Algorithme **KNN** (K-Nearest Neighbors) pour identifier les images les plus proches d'une requête.
* **Interface Web Interactive :**
    * Dashboard complet permettant de configurer l'analyse, visualiser les courbes de performance et explorer les résultats.

---

## 🛠️ Stack Technique

Ce projet a été réalisé en utilisant le langage **Python** et les bibliothèques suivantes :

| Domaine | Technologies Utilisées |
| :--- | :--- |
| **Backend Web** | `Flask` (API REST) |
| **Frontend** | `HTML5`, `CSS3`, `JavaScript`, `Chart.js` (Visualisation) |
| **Traitement d'Image** | `OpenCV` (cv2), `Mahotas` |
| **Machine Learning** | `Scikit-learn` (K-Means, KNN) |
| **Deep Learning** | `TensorFlow` / `Keras` (Modèle VGG19) |
| **Calcul & Données** | `NumPy`, `Matplotlib` |

---

## 📂 Base de Données

Le système a été entraîné et testé sur une base de données organisée en **12 classes** distinctes, totalisant environ 1200 images :

* 🚲 Bike (Vélo)
* 🚌 Bus
* 🐱 Cat (Chat)
* 🦖 Dinosaurs (Dinosaures)
* 🐶 Dog (Chien)
* 🐘 Elephants (Éléphants)
* 🌸 Flowers (Fleurs)
* 🍔 Foods (Nourriture)
* 🐎 Horses (Chevaux)
* 🏛️ Monuments
* 🏔️ Mountains & Snow (Montagnes)
* ⌚ Watch (Montres)

---

## 🚀 Installation et Utilisation

### Prérequis
* Python 3.8+
* pip

### 1. Cloner le projet
```bash
git clone [https://github.com/VOTRE_USERNAME/NOM_DU_PROJET.git](https://github.com/VOTRE_USERNAME/NOM_DU_PROJET.git)
cd NOM_DU_PROJET

# 🔍 Recherche d'Images Similaires & Clustering (CBIR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-VGG19-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge&logo=opencv&logoColor=white)

> **Projet de Machine Learning & Vision par Ordinateur**
> Une application web complète pour l'analyse, le clustering et la recherche d'images similaires, comparant des descripteurs classiques (Couleur, Forme, Texture) et le Deep Learning (VGG19).

---

## 📝 Description du Projet

[cite_start]Ce projet implémente un pipeline de traitement d'images capable de structurer une base de données visuelle et de retrouver des images similaires à une requête (CBIR - Content-Based Image Retrieval). [cite: 10, 26]

L'objectif principal est de comparer l'efficacité des méthodes d'extraction de caractéristiques traditionnelles face aux réseaux de neurones convolutionnels (CNN) via une interface web interactive.

### ✨ Fonctionnalités Clés

* [cite_start]**Extraction de caractéristiques (Feature Extraction) :** [cite: 26]
    * [cite_start]🎨 **Couleur :** Histogramme RGB (Distribution chromatique)[cite: 97].
    * [cite_start]📐 **Forme :** Moments de Zernike (Invariants par rotation)[cite: 26, 120].
    * [cite_start]🧱 **Texture :** SFTA (Segmentation-based Fractal Texture Analysis)[cite: 26, 154].
    * [cite_start]🧠 **Deep Learning :** Features extraites via **VGG19** (Transfer Learning)[cite: 26, 571].
* **Clustering Non-Supervisé :**
    * [cite_start]Utilisation de l'algorithme **K-Means** pour grouper les images[cite: 28, 626].
    * [cite_start]Optimisation du nombre de clusters ($k$) via la **méthode du Coude (Elbow Method)**[cite: 28, 679].
    * [cite_start]Évaluation de la qualité des clusters via le **Score de Silhouette**[cite: 28, 755].
* **Recherche de Similarité :**
    * [cite_start]Algorithme **KNN** (K-Nearest Neighbors) pour identifier les images les plus proches d'une requête[cite: 28, 840].
* **Interface Web Interactive :**
    * [cite_start]Dashboard complet permettant de configurer l'analyse, visualiser les courbes de performance et explorer les résultats[cite: 911].

---

## 🛠️ Stack Technique

[cite_start]Ce projet a été réalisé en utilisant le langage **Python** et les bibliothèques suivantes : [cite: 356, 366]

| Domaine | Technologies Utilisées |
| :--- | :--- |
| **Backend Web** | [cite_start]`Flask` (API REST) [cite: 877] |
| **Frontend** | [cite_start]`HTML5`, `CSS3`, `JavaScript`, `Chart.js` (Visualisation) [cite: 877, 894] |
| **Traitement d'Image** | [cite_start]`OpenCV` (cv2), `Mahotas` [cite: 367, 372] |
| **Machine Learning** | [cite_start]`Scikit-learn` (K-Means, KNN) [cite: 371] |
| **Deep Learning** | [cite_start]`TensorFlow` / `Keras` (Modèle VGG19) [cite: 373] |
| **Calcul & Données** | [cite_start]`NumPy`, `Matplotlib` [cite: 368, 369] |

---

## 📂 Base de Données

[cite_start]Le système a été entraîné et testé sur une base de données organisée en **12 classes** distinctes, totalisant environ 1200 images : [cite: 295, 298]

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

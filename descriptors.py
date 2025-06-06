# descriptors.py
import os
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mahotas as mt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import Counter

# ―― Descripteurs classiques ――

class Color:
    def histogram(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros(512)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        return cv2.normalize(hist, hist).flatten()

class Shape:
    def zernike_moments(self, path, radius=21, degree=8):
        img = cv2.imread(path)
        if img is None:
            return np.zeros(45)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        _, b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return mt.features.zernike_moments(b, radius, degree)

class Texture:
    def sfta(self, path, num_thresh=8):
        img = cv2.imread(path)
        if img is None:
            return np.zeros(num_thresh * 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = []
        thr = np.linspace(0, 255, num_thresh + 1)[1:]
        for t in thr:
            b = np.zeros_like(gray)
            b[gray > t] = 255
            ctrs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = sum(cv2.contourArea(c) for c in ctrs)
            peri = sum(cv2.arcLength(c, True) for c in ctrs)
            white = np.sum(b > 0)
            feats.extend([area, peri, white])
        return np.array(feats)

# ―― Pipeline couleur + forme + texture + VGG19 ――

class CombinedFeatureExtractor:
    def __init__(self, weights=None, n_clusters=9, n_neighbors=11):
        # Poids par défaut si non fournis
        if weights is None:
            self.weights = {
                'color':   0.20,
                'shape':   0.15,
                'texture': 0.20,
                'vgg':     0.45
            }
        else:
            self.weights = weights

        # Instanciation des extracteurs classiques
        self.color   = Color()
        self.shape   = Shape()
        self.texture = Texture()

        # Modèle VGG19 sans têtes, pooling global avg
        base = VGG19(weights='imagenet', include_top=False, pooling='avg')
        self.vgg_model = Model(inputs=base.input, outputs=base.output)

        # Clustering & KNN (initialement non entraînés)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.knn    = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')

    def extract_features(self, path):
        # 1) Descripteurs classiques
        c = self.color.histogram(path)
        s = self.shape.zernike_moments(path)
        t = self.texture.sfta(path)

        # 2) Descripteur deep (VGG19)
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img)[None, ...]
        arr = preprocess_input(arr)
        v = self.vgg_model.predict(arr, verbose=0).flatten()

        # 3) Normalisation l2
        def l2(x):
            return x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x
        c, s, t, v = map(l2, (c, s, t, v))

        # 4) Concaténation pondérée
        feats = np.concatenate([
            c * self.weights['color'],
            s * self.weights['shape'],
            t * self.weights['texture'],
            v * self.weights['vgg']
        ])
        return feats.astype(np.float64)

    def compute_vgg_descriptor(self, path):
        """
        Renvoie uniquement le vecteur VGG brut (avant pondération),
        pour usage direct dans l'API Flask.
        """
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img)[None, ...]
        arr = preprocess_input(arr)
        v = self.vgg_model.predict(arr, verbose=0).flatten()
        return v

    def get_image_files(self, root):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = []
        for r, _, names in os.walk(root):
            for f in names:
                if f.lower().endswith(exts):
                    files.append(os.path.join(r, f))
        return files

    def process_directory(self, root):
        files = self.get_image_files(root)
        feats, paths, labels = [], [], []
        for p in files:
            feats.append(self.extract_features(p))
            paths.append(p)
            labels.append(os.path.basename(os.path.dirname(p)))
        X = np.vstack(feats)
        clusters = self.kmeans.fit_predict(X)
        self.knn.fit(X)
        self._X = X  # Pour silhouette, etc.
        df = pd.DataFrame({
            'ImagePath':  paths,
            'ClassLabel': labels,
            'cluster':    clusters
        })
        return df

    def find_similar_and_classify(self, query, df, k=11):
        """
        1) Extrait les features de l'image requête
        2) Recherche ses k plus proches voisins dans l'espace des features
        3) Vote majoritaire sur les clusters des voisins pour déterminer le cluster de la requête
        4) Retourne :
           - sims           : liste de dicts {path, class, cluster, distance, filename}
           - q_cluster      : cluster prédiction par vote majoritaire
           - pred_knn_class : classe prédite par vote majoritaire
        """
        # 1) Extraction des features de la requête
        qf = self.extract_features(query)[None, :]

        # 2) Recherche des k plus proches voisins
        dists, idxs = self.knn.kneighbors(qf, n_neighbors=k)

        # 3) Vote majoritaire sur les clusters des voisins
        neighbor_clusters = [int(df.iloc[i]['cluster']) for i in idxs.flatten()]
        q_cluster = Counter(neighbor_clusters).most_common(1)[0][0] if neighbor_clusters else None

        # 4) Construction de la liste “sims”
        sims = []
        for dist, i in zip(dists.flatten(), idxs.flatten()):
            row = df.iloc[i]
            sims.append({
                'path':     row['ImagePath'],
                'class':    row['ClassLabel'],
                'cluster':  int(row['cluster']),
                'distance': float(dist),
                'filename': os.path.basename(row['ImagePath'])
            })

        # 5) Vote majoritaire sur la classe KNN
        classes_voisins = [r['class'] for r in sims]
        pred_knn = Counter(classes_voisins).most_common(1)[0][0] if classes_voisins else None

        return sims, q_cluster, pred_knn


# ―― Fonctions de visualisation (optionnelles en mode CLI) ――

def plot_elbow(X, k_min=2, k_max=14):
    Ks, inertias = range(k_min, k_max + 1), []
    for k in Ks:
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(Ks, inertias, marker='o')
    plt.title("Elbow Curve (inertie intra-classe)")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.xticks(Ks); plt.tight_layout(); plt.show()

def plot_silhouette(X, labels):
    n_clusters = len(np.unique(labels))
    sil_vals = silhouette_samples(X, labels)
    sil_avg = silhouette_score(X, labels)
    y_lower = 10
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        vals = np.sort(sil_vals[labels == i])
        y_upper = y_lower + len(vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * len(vals), str(i))
        y_lower = y_upper + 10
    plt.axvline(x=sil_avg, color="red", linestyle="--")
    plt.title(f"Silhouette plot (avg={sil_avg:.3f})")
    plt.xlabel("Coefficient de silhouette"); plt.ylabel("Cluster")
    plt.yticks([]); plt.tight_layout(); plt.show()

def plot_cluster_distribution(df):
    counts = df['cluster'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind='bar')
    plt.title("Distribution par cluster")
    plt.xlabel("Cluster"); plt.ylabel("Count")
    plt.tight_layout(); plt.show()

def plot_cluster_samples(df, samples_per_cluster=3):
    clusters = sorted(df['cluster'].unique())
    per_row, cols = 3, 3 * samples_per_cluster
    rows = math.ceil(len(clusters) / per_row)
    plt.figure(figsize=(15, rows * 5))
    for idx, cl in enumerate(clusters):
        r_blk, c_blk = divmod(idx, per_row)
        subs = df[df['cluster'] == cl].sample(min(samples_per_cluster, len(df[df['cluster'] == cl])))
        for j, (_, r) in enumerate(subs.iterrows()):
            pos = r_blk * cols + c_blk * samples_per_cluster + (j + 1)
            ax = plt.subplot(rows, cols, pos)
            img = cv2.imread(r['ImagePath'])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((100, 100, 3), dtype=np.uint8)
            ax.imshow(img); ax.set_title(f"Cluster {cl}"); ax.axis('off')
    plt.tight_layout(); plt.show()

def plot_similar_images(query, sims, q_cluster, pred_knn):
    print(f"\nRequête : {os.path.basename(query)}")
    print(f"→ Cluster KMeans (vote majoritaire) : {q_cluster}")
    print(f"→ Classe prédite KNN            : {pred_knn}")
    correct11 = sum(1 for r in sims[:11] if r['class'] == pred_knn)
    print(f"KNN Accuracy@11      : {correct11}/11 = {correct11/11:.2%}\n")
    for i, r in enumerate(sims[:11], 1):
        print(f"{i}. {r['filename']} | Cl:{r['cluster']} | Cls:{r['class']} | Dist:{r['distance']:.3f}")
    paths  = [query] + [r['path'] for r in sims[:11]]
    titles = ["Requête"] + [f"{i+1}: {r['class']}\nDist {r['distance']:.2f}" for i, r in enumerate(sims[:11])]
    plt.figure(figsize=(18, 6))
    for idx, (p, t) in enumerate(zip(paths, titles)):
        ax = plt.subplot(2, 6, idx + 1)
        im = cv2.imread(p)
        if im is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = np.zeros((100, 100, 3), dtype=np.uint8)
        ax.imshow(im); ax.set_title(t, fontsize=8); ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

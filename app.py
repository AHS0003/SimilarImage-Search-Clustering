import os
import io
import math
import uuid
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend non graphique pour matplotlib
import matplotlib.pyplot as plt
import mahotas as mt

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from collections import Counter

from descriptors import CombinedFeatureExtractor

app = Flask(__name__)
app.secret_key = 'change_this_secret_for_production'

# Dossier où seront stockées temporairement les images générées (elbow, silhouette, requête)
app.config['UPLOAD_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Contiendra le dossier du dataset pour servir les vignettes
app.config['CURRENT_IMG_DIR'] = None

# Extensions autorisées pour le dataset et l'image requête
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    """
    Sert une image située dans CURRENT_IMG_DIR.
    Exemple d'URL générée: /dataset/classA/img001.jpg
    """
    img_dir = app.config.get('CURRENT_IMG_DIR', None)
    if img_dir is None or not os.path.isdir(img_dir):
        return "Dataset non disponible", 404
    return send_from_directory(img_dir, filename)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # ── 1) Récupérer les descripteurs sélectionnés + poids ──
    selected = request.form.getlist('descriptor')
    if not selected:
        flash("Vous devez sélectionner au moins un descripteur.", "danger")
        return redirect(url_for('index'))

    weights = {}
    for desc in selected:
        try:
            w = float(request.form.get(f'weight_{desc}', 1.0))
        except ValueError:
            w = 1.0
        weights[desc] = w

    # ── 2) Récupérer k et n_neighbors ──
    try:
        k = int(request.form.get('k_clusters', 3))
        n_neighbors = int(request.form.get('n_neighbors', 5))
    except ValueError:
        flash("k_clusters et n_neighbors doivent être des entiers.", "danger")
        return redirect(url_for('index'))

    # ── 3) Récupérer le chemin du dossier d'images ──
    img_dir = request.form.get('img_dir', '').strip().strip('"').strip("'")
    if img_dir == '' or not os.path.isdir(img_dir):
        flash(f"Dossier d'images invalide ou introuvable : {img_dir}", "danger")
        return redirect(url_for('index'))

    # Enregistrer le dossier pour servir les vignettes
    app.config['CURRENT_IMG_DIR'] = img_dir

    # ── 4) Récupérer l'image requête ──
    if 'query_image' not in request.files:
        flash("Le champ 'query_image' est manquant.", "danger")
        return redirect(url_for('index'))

    query_file = request.files['query_image']
    if query_file.filename == '':
        flash("Aucune image requête sélectionnée.", "danger")
        return redirect(url_for('index'))
    if not allowed_file(query_file.filename):
        flash("Format d'image requête non supporté.", "danger")
        return redirect(url_for('index'))

    # Générer un nom unique pour l'image requête
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(query_file.filename)}"
    query_save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    query_file.save(query_save_path)

    # ── 5) Lister tous les chemins d'images valides dans img_dir ──
    img_paths_abs = []
    for root, dirs, files in os.walk(img_dir):
        for fname in files:
            if allowed_file(fname):
                img_paths_abs.append(os.path.join(root, fname))
    img_paths_abs.sort()
    if not img_paths_abs:
        flash(f"Pas d'images valides trouvées dans le dossier : {img_dir}", "warning")
        return redirect(url_for('index'))

    # Vérifier que k ne dépasse pas le nombre d'images
    if k > len(img_paths_abs):
        flash(f"Le nombre de clusters k={k} ne peut pas être supérieur au nombre d'images ({len(img_paths_abs)})", "danger")
        return redirect(url_for('index'))

    # ── 6) Initialiser l'extracteur avec les poids et paramètres ──
    extractor = CombinedFeatureExtractor(weights=weights, n_clusters=k, n_neighbors=n_neighbors)

    # ── 7) Calculer les descripteurs pour chaque image du dataset ──
    features_dict = {}
    for desc in selected:
        feats_list = []
        for path in img_paths_abs:
            if desc == 'color':
                vec = extractor.color.histogram(path)
            elif desc == 'shape':
                vec = extractor.shape.zernike_moments(path)
            elif desc == 'texture':
                vec = extractor.texture.sfta(path)
            elif desc == 'vgg':
                vec = extractor.compute_vgg_descriptor(path)
            else:
                continue
            if vec is None or not isinstance(vec, np.ndarray):
                vec = np.zeros((1,), dtype=np.float32)
            feats_list.append(weights[desc] * vec)
        feats = np.stack(feats_list, axis=0)  # shape: (n_images, dim_desc)
        features_dict[desc] = feats

    # Vérifier cohérence des dimensions
    for desc, arr in features_dict.items():
        if arr.ndim != 2 or arr.shape[0] != len(img_paths_abs):
            flash(f"Erreur interne: descripteur '{desc}' de dimension inattendue {arr.shape}", "danger")
            return redirect(url_for('index'))

    # ── 8) Concaténer tous les vecteurs en X (n_images, dim_totale) ──
    X = np.concatenate([features_dict[d] for d in selected], axis=1)

    # ── 9) Tracer la courbe du coude ──
    inertia_vals = []
    # Commencer à 2 pour éviter k=1
    for kv in range(2, k + 1):
        km_temp = KMeans(n_clusters=kv, random_state=42, n_init=10)
        km_temp.fit(X)
        inertia_vals.append(km_temp.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, k + 1), inertia_vals, 'bo-', linewidth=2)
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Inertie (intra-classe)")
    plt.title("Courbe du coude")
    elbow_file = f"elbow_{uuid.uuid4().hex}.png"
    elbow_path = os.path.join(app.config['UPLOAD_FOLDER'], elbow_file)
    plt.savefig(elbow_path)
    plt.close()

    # ── 10) Appliquer KMeans final ──
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # ── 11) Calculer score & graphique de silhouette (si k>=2) ──
    silhouette_avg = None
    silhouette_file = None
    if k >= 2:
        silhouette_avg = silhouette_score(X, cluster_labels)
        sil_vals = silhouette_samples(X, cluster_labels)
        plt.figure(figsize=(6, 6))
        y_lower = 10
        for i in range(k):
            ith_vals = np.sort(sil_vals[cluster_labels == i])
            size_i = ith_vals.shape[0]
            y_upper = y_lower + size_i
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_vals,
                              facecolor=plt.cm.nipy_spectral(float(i)/k),
                              edgecolor=plt.cm.nipy_spectral(float(i)/k), alpha=0.7)
            plt.text(-0.05, y_lower + 0.5*size_i, str(i))
            y_lower = y_upper + 10
        plt.xlabel("Valeur de silhouette")
        plt.ylabel("Cluster")
        plt.title(f"Silhouette pour k={k} (moyenne={silhouette_avg:.3f})")
        silhouette_file = f"silhouette_{uuid.uuid4().hex}.png"
        silhouette_path = os.path.join(app.config['UPLOAD_FOLDER'], silhouette_file)
        plt.savefig(silhouette_path)
        plt.close()

    # ── 12) Sélection d'exemples visuels par cluster (3 images max) ──
    samples_per_cluster = {}
    for cid in range(k):
        idxs = np.where(cluster_labels == cid)[0]
        chosen = idxs[:3] if len(idxs) >= 3 else idxs
        rels = []
        for j in chosen:
            abs_p = img_paths_abs[j]
            rel_p = os.path.relpath(abs_p, img_dir).replace("\\", "/")
            rels.append(rel_p)
        samples_per_cluster[cid] = rels

    # ── 13) KNN sur l'image requête ──
    # Réentraîner l’objet knn dans l’extracteur
    extractor.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    extractor.knn.fit(X)

    query_feats = []
    for desc in selected:
        if desc == 'color':
            vq = extractor.color.histogram(query_save_path)
        elif desc == 'shape':
            vq = extractor.shape.zernike_moments(query_save_path)
        elif desc == 'texture':
            vq = extractor.texture.sfta(query_save_path)
        elif desc == 'vgg':
            vq = extractor.compute_vgg_descriptor(query_save_path)
        else:
            continue
        if vq is None or not isinstance(vq, np.ndarray):
            vq = np.zeros((1,), dtype=np.float32)
        query_feats.append(weights[desc] * vq)
    query_vector = np.hstack(query_feats).reshape(1, -1)

    distances, indices = extractor.knn.kneighbors(query_vector)
    idxs = indices.flatten().tolist()
    dists = distances.flatten().tolist()

    # Construire la liste knn_results avec cluster et classe
    knn_results = []
    neighbor_clusters = []
    labels = [os.path.basename(os.path.dirname(p)) for p in img_paths_abs]
    for i, idx in enumerate(idxs):
        abs_p = img_paths_abs[idx]
        rel_p = os.path.relpath(abs_p, img_dir).replace("\\", "/")
        knn_results.append({
            'rel_path': rel_p,
            'distance': float(dists[i]),
            'cluster':  int(cluster_labels[idx]),
            'class':    labels[idx]
        })
        neighbor_clusters.append(int(cluster_labels[idx]))

    # Vote majoritaire sur les clusters des voisins
    pred_knn_cluster = None
    if neighbor_clusters:
        pred_knn_cluster = Counter(neighbor_clusters).most_common(1)[0][0]

    # ── 14) Préparer le contexte pour la template ──
    context = {
        'elbow_image':       url_for('static', filename=f"outputs/{elbow_file}"),
        'silhouette_image':  url_for('static', filename=f"outputs/{silhouette_file}") if silhouette_file else None,
        'silhouette_score':  f"{silhouette_avg:.3f}" if silhouette_avg is not None else "N/A",
        'samples_per_cluster': samples_per_cluster,
        'knn_results':       knn_results,
        'pred_knn_cluster':  pred_knn_cluster
    }

    return render_template('results.html', **context)


if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=5000)

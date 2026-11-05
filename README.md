Les mesures internes nécessitent entre 2 et N−1 groupes distincts. ARI/NMI nécessitent des classes connues. Les scores indéfinis restent indisponibles.

Pour une requête, soit $r_i\in\{0,1\}$ la pertinence du résultat de rang $i$ et $R$ le nombre de pertinents dans la galerie :

$$
P@k=\frac{\sum_{i=1}^k r_i}{k},\qquad
Recall@k=\frac{\sum_{i=1}^k r_i}{R},
$$

$$
AP@10=
\frac{\sum_{i=1}^{10}(P@i)r_i}{\min(R,10)}.
$$

Les rangs non retournés comptent comme non pertinents. Le projet rapporte **Precision@5, Precision@10, Recall@10 et mAP@10**, moyenne de AP@10 sur les requêtes évaluables. Si $R=0$, la requête est exclue des moyennes mais comptée dans la couverture.

| Expérience | Configuration |
|---|---|
| E1 | RGB |
| E2 | Zernike |
| E3 | SFTA |
| E4 | RGB + Zernike |
| E5 | RGB + SFTA |
| E6 | Zernike + SFTA |
| E7 | RGB + Zernike + SFTA |

## 9. Installation et utilisation

Prérequis : **Python 3.12, 64 bits**.

~~~bash
git clone https://github.com/AHS0003/SimilarImage-Search-Clustering.git
cd SimilarImage-Search-Clustering
# Si le refactor n’est pas encore fusionné :
git switch feature/classical-image-retrieval-evaluation
python -m venv .venv
~~~

~~~powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
# Linux / macOS
source .venv/bin/activate
~~~

~~~bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python -m ruff check .
~~~

Les dépendances d’exécution sont dans <code>requirements.txt</code> ; pytest et Ruff sont ajoutés par <code>requirements-dev.txt</code>. Aucun modèle préentraîné n’est téléchargé.

Organiser les images dans des sous-dossiers de classe, par exemple <code>dataset/Bike/image001.jpg</code> et <code>dataset/Cat/image001.png</code>.

~~~bash
# Les sept configurations et les deux stratégies
python -m experiments.compare_descriptors --dataset /chemin/dataset --output results/comparison --seed 42 --clusters auto --strategy both

# Clustering seul, K manuel
python -m experiments.evaluate_clustering --dataset /chemin/dataset --output results/clustering --descriptors color texture --clusters 3 --k-values 2 3 4

# Recherche seule
python -m experiments.evaluate_retrieval --dataset /chemin/dataset --output results/retrieval --descriptors color shape texture --strategy both
~~~

Les K explicites doivent être valides pour la galerie. La grille par défaut va de 2 à min(10,N), avec gestion des très petites galeries.

L’option <code>--weights COLOR SHAPE TEXTURE</code> fixe les poids, par défaut 1,1,1. Comparer les sept configurations nécessite trois poids strictement positifs, pour que chaque singleton soit valide. Choisir un dossier de sortie neuf : les rapports existants ne sont pas écrasés.

Sans dataset réel :

~~~bash
python -m experiments.generate_synthetic --output datasets/synthetic --seed 42 --per-class 10
python -m experiments.compare_descriptors --dataset datasets/synthetic --output results/synthetic-smoke --seed 42 --k-values 2 3 4 --dataset-kind synthetic
~~~

Ces images simples valident le fonctionnement, pas les performances réelles.

### Sorties

| Fichier | Contenu |
|---|---|
| <code>split_manifest.csv</code> | Chemins, labels, hashes, partitions |
| <code>descriptor_times.csv</code> | Extraction par famille et décodage partagé |
| <code>clustering_metrics.csv</code> | Diagnostics pour chaque K |
| <code>gallery_clusters.csv</code> | Affectations de la galerie |
| <code>retrieval_metrics.csv</code> | Métriques agrégées |
| <code>retrieval_per_query.csv</code> | Résultats par requête |
| <code>rankings.csv</code> | Voisins, distances ou scores RRF |
| <code>clustering.png</code>, <code>retrieval.png</code> | Graphiques calculés |
| <code>metadata.json</code> | Paramètres, exclusions, seed et conventions |

Les temps d’extraction par configuration sont des sommes des temps partagés, pas sept extractions indépendantes. Les durées murales varient selon le matériel.

## 10. Tests, limites et références

### Tests

La suite vérifie les dimensions et valeurs finies, les tailles et modes d’images, les corruptions, les cas uniformes, EXIF/transparence, la rotation Zernike, le box-counting, les masques SFTA, la normalisation, les poids, les métriques, K-Means, les voisins et la reproductibilité. GitHub Actions exécute les tests, le lint et le contrôle des dépendances.

### Limites scientifiques

- Aucun résultat sur le dataset réel n’est inventé.
- SHA256 ne détecte pas les réencodages ni les quasi-doublons visuels.
- Les descripteurs restent sensibles au fond, au cadrage et à la segmentation.
- L2 par famille ne garantit ni indépendance des composantes ni équilibre interne parfait.
- Un seed fixe rend un passage reproductible, mais ne quantifie pas l’incertitude : une étude complète devrait comparer plusieurs partitions.
- Le K maximisant la silhouette ne représente pas nécessairement le nombre de classes sémantiques.

### Références

- Costa, Humpire-Mamani et Traina, *An Efficient Algorithm for Fractal Analysis of Textures*, SIBGRAPI 2012 : [publication](https://doi.org/10.1109/SIBGRAPI.2012.15).
- [Code SFTA de l’auteur](https://github.com/alceufc/sfta/blob/master/matlab/sfta.m), avec les fonctions de seuils, frontières et dimension fractale du même dépôt.
- [Implémentation des moments de Zernike dans Mahotas](https://github.com/luispedro/mahotas/blob/main/mahotas/features/zernike.py).
- [Histogrammes OpenCV](https://docs.opencv.org/4.x/d8/dbc/tutorial_histogram_calculation.html).
- [Évaluation du clustering scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation).

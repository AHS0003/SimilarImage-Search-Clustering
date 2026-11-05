# Classical Image Retrieval & Clustering

Application Python de recherche d’images par le contenu (**CBIR**) et de clustering. Le projet compare trois descripteurs classiques : **histogramme RGB**, **moments de Zernike** et **SFTA**, seuls ou combinés.

Cette version expérimentale utilise K-Means et les voisins euclidiens. Flask, VGG19 et TensorFlow ont été retirés.

## Sommaire

1. Objectif et architecture
2. Prétraitement
3. Histogramme RGB
4. Moments de Zernike
5. SFTA
6. Normalisation et fusion
7. Clustering et recherche
8. Évaluation
9. Installation et utilisation
10. Tests, limites et références

## 1. Objectif et architecture

Une image est transformée en un vecteur numérique. Les distances entre ces vecteurs permettent de regrouper les images et de retrouver les plus similaires à une requête.

| Descripteur | Information représentée | Dimension par défaut |
|---|---|---:|
| RGB | Distribution conjointe des couleurs | 512 |
| Zernike | Géométrie du premier plan segmenté | 25 |
| SFTA | Frontières issues de la segmentation des textures | 45 |
| Fusion des trois | Couleur + forme + texture | **582** |

~~~text
src/
├── preprocessing.py     # Décodage et redimensionnement
├── descriptors.py       # RGB, Zernike, SFTA, normalisation, fusion
├── clustering.py        # K-Means, sélection de K, diagnostics
├── retrieval.py         # Voisins euclidiens et fusion des rangs
├── metrics.py           # Mesures de recherche
└── evaluation.py        # Partitions, expériences, exports
experiments/
├── compare_descriptors.py
├── evaluate_clustering.py
├── evaluate_retrieval.py
└── generate_synthetic.py
tests/
results/
.github/workflows/tests.yml
~~~

## 2. Prétraitement

On note l’image couleur $I:\Omega\rightarrow\{0,\ldots,255\}^3$, où $\Omega$ désigne ses pixels valides.

Le décodage applique l’orientation EXIF, convertit l’image en RGB et compose la transparence sur blanc. Pour un GIF, seule la première image est utilisée. Une requête illisible est refusée ; les images corrompues du dataset sont exclues et signalées.

Pour une image de largeur $W$ et de hauteur $H$, le facteur de redimensionnement vers un plus grand côté $L$ est :

$$
s=\frac{L}{\max(W,H)},\qquad
W'=\max(1,\operatorname{round}(sW)),\quad
H'=\max(1,\operatorname{round}(sH)).
$$

Les proportions sont conservées à l’arrondi près.

| Famille | Prétraitement |
|---|---|
| RGB | Ajustement à 256×256 avec padding exclu des statistiques |
| Zernike | Segmentation puis ajustement proportionnel dans un support 128×128 |
| SFTA | Image grise dont le plus grand côté vaut 256, sans padding ajouté aux statistiques |

**Redimensionnement et normalisation sont distincts :** le premier agit sur les pixels ; la seconde agit sur les vecteurs extraits.

## 3. Histogramme RGB : décrire la couleur

Implémentation : <code>Color.histogram()</code> dans <code>src/descriptors.py</code>.

### 3.1 Quantification

Chaque canal est découpé en huit intervalles de largeur 32 :

$$
q(v)=\left\lfloor\frac{8v}{256}\right\rfloor
=\left\lfloor\frac{v}{32}\right\rfloor,\qquad v\in\{0,\ldots,255\}.
$$

La valeur quantifiée appartient à $\{0,\ldots,7\}$. Par exemple, 0 à 31 appartiennent au bin 0 et 224 à 255 au bin 7.

### 3.2 Histogramme conjoint

Pour chaque combinaison $(i,j,k)$ des trois canaux :

$$
H(i,j,k)=
\sum_{p\in\Omega}
\mathbf{1}[q(R_p)=i,\ q(G_p)=j,\ q(B_p)=k].
$$

L’indicatrice $\mathbf{1}$ vaut 1 si la condition est satisfaite, 0 sinon. Les pixels artificiels de padding ne sont pas comptés.

L’aplatissement du tableau produit :

$$
\mathbf h_{RGB}=\operatorname{vec}(H)\in\mathbb R^{8^3}
=\mathbb R^{512}.
$$

Il s’agit d’un histogramme **conjoint à 512 composantes**, et non de trois histogrammes indépendants de huit composantes, qui n’en produiraient que 24.

Exemple : le pixel $(240,30,20)$ incrémente le bin $(7,0,0)$.

### 3.3 Interprétation et limites

Ce descripteur représente la fréquence des couleurs, mais ignore leur position. Deux images contenant les mêmes proportions de rouge et de bleu peuvent donc avoir le même histogramme malgré des compositions différentes.

Il est sensible à l’éclairage, au fond et aux changements colorimétriques. Le rééchantillonnage peut aussi déplacer légèrement les valeurs entre bins. La méthode renvoie des comptages bruts ; la normalisation L2 intervient dans le pipeline commun.

## 4. Moments de Zernike : décrire la forme

Implémentation : <code>Shape.zernike_moments()</code>, avec Mahotas.

### 4.1 Segmentation du premier plan

L’image est convertie en niveaux de gris. Otsu choisit un seuil maximisant la variance interclasse :

$$
t^\star=\arg\max_t
\omega_0(t)\omega_1(t)
[\mu_0(t)-\mu_1(t)]^2.
$$

$\omega_0,\omega_1$ sont les proportions des deux groupes et $\mu_0,\mu_1$ leurs intensités moyennes.

Le masque est inversé si la majorité de son bord est blanche : c’est une heuristique de sélection du premier plan. Celui-ci est recadré, redimensionné proportionnellement puis placé dans un canevas noir de 128×128.

### 4.2 Centrage et disque unité

Pour les $N_f$ pixels du premier plan :

$$
c_x=\frac1{N_f}\sum_p x_p,\qquad
c_y=\frac1{N_f}\sum_p y_p.
$$

Le rayon automatique entoure tout le premier plan :

$$
r=\max\left(1,\max_p\sqrt{(x_p-c_x)^2+(y_p-c_y)^2}+\varepsilon\right).
$$

On utilise ensuite :

$$
x'_p=\frac{x_p-c_x}{r},\quad
y'_p=\frac{y_p-c_y}{r},\quad
\rho_p=\sqrt{x_p'^2+y_p'^2},\quad
\theta_p=\operatorname{atan2}(y'_p,x'_p).
$$

Le calcul s’effectue sur le disque $\rho\leq1$. Mahotas attend le centre dans l’ordre **ligne, colonne**, soit $(c_y,c_x)$.

### 4.3 Polynômes de Zernike

Les fonctions de base sont :

$$
V_n^m(\rho,\theta)=R_n^m(\rho)e^{im\theta},
$$

avec $n\geq0$, $|m|\leq n$ et $n-|m|$ pair. Leur partie radiale est :

$$
R_n^m(\rho)=
\sum_{s=0}^{(n-|m|)/2}
(-1)^s
\frac{(n-s)!}
{s!\left(\frac{n+|m|}{2}-s\right)!
\left(\frac{n-|m|}{2}-s\right)!}
\rho^{n-2s}.
$$

$n$ contrôle l’ordre radial ; $m$ la variation angulaire.

### 4.4 Moments et invariance à la rotation

La définition continue usuelle est :

$$
A_n^m=\frac{n+1}{\pi}
\iint_{x^2+y^2\leq1}
f(x,y)\overline{V_n^m(x,y)}\,dx\,dy.
$$

La barre représente le conjugué complexe. Dans le calcul discret, Mahotas normalise les intensités positives par leur somme. Avec ces poids $\widetilde f_p$, de somme 1 :

$$
A_n^m=
\frac{n+1}{\pi}
\sum_{\rho_p\leq1}
\widetilde f_p R_n^m(\rho_p)e^{-im\theta_p}.
$$

Le descripteur conserve les modules $|A_n^m|$.

Pour une rotation idéale d’angle $\alpha$ :

$$
A_n^{m\,\prime}=e^{-im\alpha}A_n^m
\quad\Longrightarrow\quad
|A_n^{m\,\prime}|=|A_n^m|.
$$

La phase change, mais pas le module. Sur une image numérique, cette invariance est approximative à cause de l’interpolation et de la segmentation. Le centrage et le rayon adapté réduisent également la dépendance à la translation et à l’échelle.

### 4.5 Dimension exacte

Mahotas conserve les indices $m\geq0$ respectant la parité. Au degré maximal $d$ :

$$
D_Z(d)=\sum_{n=0}^{d}
\left(\left\lfloor\frac n2\right\rfloor+1\right).
$$

Pour $d=8$ :

$$
D_Z(8)=1+1+2+2+3+3+4+4+5=\boxed{25}.
$$

Une image uniforme, sans premier plan identifiable, produit donc 25 zéros. La méthode dépend néanmoins de la qualité du masque et ne remplace pas une segmentation sémantique.

## 5. SFTA : décrire la texture

Implémentation : <code>Texture.sfta()</code>.

**Segmentation-based Fractal Texture Analysis** décompose une image grise en masques binaires et caractérise leurs frontières. Le projet suit les statistiques de frontière du code de référence de Costa et collaborateurs.

### 5.1 Seuils récursifs

Pour l’image grise $G:\Omega\rightarrow\{0,\ldots,255\}$, Otsu est appliqué récursivement aux intervalles de l’histogramme. On obtient :

$$
T=(t_1,\ldots,t_{n_t}),\qquad t_1\leq\cdots\leq t_{n_t}.
$$

Ce procédé récursif n’est pas une recherche exhaustive du meilleur ensemble global de seuils. Dans les branches uniformes ou vides, des seuils sont répétés pour préserver la dimension de sortie.

### 5.2 Décomposition binaire TTBD

Les masques supérieurs sont :

$$
B_i^+(p)=\mathbf{1}[G(p)>t_i],
\qquad i=1,\ldots,n_t.
$$

Les masques entre seuils adjacents sont :

$$
B_i^{int}(p)=\mathbf{1}[t_i<G(p)<t_{i+1}],
\qquad i=1,\ldots,n_t-1.
$$

Les inégalités strictes correspondent à l’implémentation. Le nombre total de masques est :

$$
N_B=n_t+(n_t-1)=2n_t-1.
$$

### 5.3 Frontières binaires

La frontière intérieure d’un masque $B$ est :

$$
\partial B=B\setminus(B\ominus S_8),
$$

où $\ominus$ est l’érosion et $S_8$ un élément structurant carré 3×3. Un pixel appartient à la frontière lorsqu’il appartient au masque et qu’au moins un de ses huit voisins en est absent.

Pour cette érosion, l’extérieur de l’image est traité comme du premier plan : le cadre ne génère pas une frontière artificielle.

### 5.4 Dimension fractale par box-counting

Une grille de carrés de côté $\epsilon$ recouvre la frontière. On compte les carrés contenant au moins un pixel :

$$
N(\epsilon)=\#\{\text{carrés occupés de côté }\epsilon\}.
$$

La dimension théorique est :

$$
D=\lim_{\epsilon\to0}
\frac{\log N(\epsilon)}{\log(1/\epsilon)}.
$$

Cette limite n’est pas observable sur une image finie. Le code estime une pente sur des tailles dyadiques :

$$
\epsilon_j\in\{1,2,4,\ldots,S\},\quad
x_j=\log(1/\epsilon_j),\quad y_j=\log N(\epsilon_j),
$$

$$
\widehat D=
\frac{\sum_j(x_j-\overline x)(y_j-\overline y)}
{\sum_j(x_j-\overline x)^2}.
$$

L’image binaire est complétée par du fond noir jusqu’à un carré de côté $S$, puissance de deux. L’estimation est bornée numériquement dans $[0,2]$. Une frontière vide ou réduite à un pixel donne 0.

Une ligne idéale a une dimension proche de 1 et une région plane remplie proche de 2. Ici, l’estimation porte sur les **frontières segmentées**, à résolution finie ; elle ne prouve pas qu’un objet possède une structure fractale.

### 5.5 Statistiques de frontière

Pour une frontière non vide :

$$
\mu_{\partial B}=\frac1{|\partial B|}
\sum_{p\in\partial B}G(p),\qquad
C_{\partial B}=|\partial B|.
$$

Chaque masque fournit le triplet :

$$
\mathbf s(B)=
[\widehat D(\partial B),\ \mu_{\partial B},\ C_{\partial B}].
$$

Il s’agit de la dimension fractale, de la moyenne des gris et du nombre de pixels de **frontière**. Ce comptage n’est pas l’aire du masque ni un périmètre continu. Une frontière vide produit $[0,0,0]$.

### 5.6 Dimension et limites

$$
D_{SFTA}=3(2n_t-1)=6n_t-3.
$$

Avec huit seuils :

$$
D_{SFTA}=6\times8-3=\boxed{45}.
$$

Les masques supérieurs sont décrits en premier, puis les intervalles. Cette taille suit le code effectif de l’auteur ; son ancien commentaire mentionnant $6n_t$ ne correspond pas à la taille calculée.

La résolution standardisée réduit la dépendance à la taille originale, sans rendre SFTA invariant au cadrage ou à tout rééchantillonnage. La normalisation du bloc entier ne standardise pas séparément ses composantes fractales, photométriques et de comptage.

## 6. Normalisation et fusion

### 6.1 L2 par famille

Pour le vecteur brut $\mathbf f_d$ :

$$
\widehat{\mathbf f}_d=
\begin{cases}
\mathbf f_d/\|\mathbf f_d\|_2,&\|\mathbf f_d\|_2>0,\\
\mathbf0,&\text{sinon},
\end{cases}
\qquad
\|\mathbf f_d\|_2=\sqrt{\sum_j f_{d,j}^2}.
$$

Cela réduit les déséquilibres dus aux amplitudes brutes des familles. Le code met d’abord les valeurs à l’échelle par leur maximum absolu pour éviter un débordement numérique, sans changer la direction normalisée.

### 6.2 Concaténation pondérée

$$
\mathbf F(I)=
[w_c\widehat{\mathbf h}_{RGB}(I),\
w_z\widehat{\mathbf z}(I),\
w_t\widehat{\mathbf s}(I)]
\in\mathbb R^{582}.
$$

Les poids sont finis, non négatifs et non tous nuls pour une configuration sélectionnée. Aucun ajustement global supplémentaire n’est effectué.

$$
\|\mathbf F(I)-\mathbf F(J)\|_2^2
=
\sum_{d\in\{c,z,t\}}w_d^2
\|\widehat{\mathbf f}_d(I)-\widehat{\mathbf f}_d(J)\|_2^2.
$$

**Doubler un poids multiplie par quatre sa contribution à la distance au carré.** La galerie et les requêtes utilisent exactement la même transformation.

Les blocs sont extraits une seule fois puis réutilisés. Aucun scaler global n’est appris à partir des requêtes.

## 7. Clustering et recherche

### K-Means

K-Means minimise l’inertie :

$$
\min_{\{C_k,\boldsymbol\mu_k\}_{k=1}^K}
\sum_{k=1}^K\sum_{\mathbf x_i\in C_k}
\|\mathbf x_i-\boldsymbol\mu_k\|_2^2,
\qquad
\boldsymbol\mu_k=\frac1{|C_k|}\sum_{\mathbf x_i\in C_k}\mathbf x_i.
$$

Le code utilise l’algorithme de Lloyd, une seed fixe et dix initialisations. Plusieurs K sont évalués. Le choix automatique maximise la silhouette finie de la **galerie**, départage les égalités par K inférieur et se replie explicitement sur K=1 si aucun candidat n’est admissible. Les candidats produisant moins de groupes distincts que demandé ne sont pas retenus automatiquement. K peut être imposé manuellement.

Un cluster appris n’est pas une classe sémantique.

### Recherche euclidienne

Pour la requête $Q$, les images sont classées par distance croissante :

$$
d(Q,I)=\|\mathbf F(Q)-\mathbf F(I)\|_2.
$$

L’index NearestNeighbors est ajusté sur la galerie uniquement. L’identité canonique de la requête est exclue des résultats ; les égalités sont départagées par le chemin canonique.

### Fusion des classements

La seconde stratégie est une fusion pondérée des rangs réciproques :

$$
\operatorname{RRF}(I\mid Q)
=
\sum_{d:w_d>0}\frac{w_d}{60+\operatorname{rang}_d(I\mid Q)}.
$$

Les rangs commencent à 1 et le score le plus grand est le meilleur. **Le score RRF n’est pas une distance euclidienne.** Cette fusion concerne la recherche, tandis que K-Means travaille sur les vecteurs concaténés.

## 8. Évaluation

Le dataset réel d’environ 1 200 images et 12 catégories n’est pas fourni.

1. Trier les chemins, dédupliquer les fichiers identiques par SHA256 et filtrer les images corrompues.
2. Construire une partition reproductible **60 % galerie / 20 % validation / 20 % test**, stratifiée si les classes existent.
3. Garder au moins trois images lisibles et uniques par classe ; l’arrondi assure au moins une image par classe dans chaque sous-ensemble.
4. Ajuster K-Means et l’index sur la galerie seulement. Les étiquettes ne servent qu’à la partition et à l’évaluation.
5. Fixer les poids avant le passage ; ne pas choisir les hyperparamètres à partir du test.

La classe est déduite du dossier parent immédiat. Si une image se trouve directement à la racine, les étiquettes sont considérées comme indisponibles.

| Mesure de clustering | Interprétation |
|---|---|
| Silhouette | Cohésion/séparation ; plus élevée préférable |
| Davies–Bouldin | Dispersion relative des groupes ; plus faible préférable |
| Calinski–Harabasz | Dispersion inter/intra ; plus élevée préférable |
| ARI | Accord avec les classes corrigé du hasard |
| NMI | Information mutuelle normalisée avec les classes |

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


# Recherche et regroupement d’images

Ce projet Python compare des images à partir de leur **couleur**, de leur **forme** et de leur **texture**. Il permet de retrouver des images similaires, de les regrouper avec **K-Means** et de comparer plusieurs méthodes.

Le programme fonctionne en ligne de commande, sans interface graphique ni modèle de deep learning.

> Le dataset réel annoncé, d’environ 1 200 images, reste à évaluer. Les essais synthétiques vérifient le fonctionnement du programme ; ils ne prouvent pas ses performances sur des images réelles.

## 1. Le principe

Chaque image est transformée en une liste de nombres appelée **vecteur de descripteurs**.

| Descripteur | Information décrite | Nombre de valeurs |
|---|---|---:|
| Histogramme RGB | Répartition des couleurs | 512 |
| Moments de Zernike | Forme du premier plan segmenté | 25 |
| SFTA | Frontières de régions obtenues à partir des niveaux de gris | 45 |
| Les trois combinés | Couleur, forme et texture | **582** |

Le nombre de valeurs du descripteur est indépendant du nombre de pixels : avec les réglages par défaut, une image de 800 × 800 pixels donne aussi 512 valeurs pour RGB.

## 2. Prétraitement : préparer les images

Un **pixel** est un point de l’image. Une image de 800 × 800 contient 800 lignes et 800 colonnes, soit 640 000 pixels.

Le programme corrige l’orientation EXIF, convertit l’image en RGB et remplace la transparence par un fond blanc. Pour un GIF, seule la première image est utilisée. Les fichiers illisibles du dataset sont exclus et signalés.

Chaque descripteur utilise ensuite une version adaptée :

| Descripteur | Préparation |
|---|---|
| RGB | Redimensionnement proportionnel dans un carré de 256 × 256, puis ajout d’une bordure si nécessaire |
| Zernike | Conversion en gris, séparation du premier plan et du fond, recadrage et placement du masque dans un support de 128 × 128 |
| SFTA | Redimensionnement proportionnel avec un plus grand côté de 256 pixels, puis conversion en gris |

La bordure ajoutée s’appelle le **padding**. Elle évite de déformer l’image et n’est pas comptée dans l’histogramme RGB.

**Exemple :** une image de 800 × 400 devient 256 × 128. Pour RGB, on ajoute 64 pixels en haut et 64 en bas afin d’obtenir 256 × 256. Une image carrée de 800 × 800 devient directement 256 × 256.

## 3. Les trois descripteurs

### RGB : compter les couleurs

Chaque pixel possède trois valeurs entre 0 et 255 : rouge, vert et bleu.

Chaque canal est découpé en **8 intervalles** de largeur 32 : de 0 à 31, de 32 à 63, et ainsi de suite jusqu’à 255.

$$
q(v)=\left\lfloor\frac{v}{32}\right\rfloor
$$

Par exemple, le pixel $(240,30,20)$ appartient à la combinaison $(7,0,0)$.

Le programme compte les pixels dans chaque combinaison. Il existe :

$$
8\times8\times8=\boxed{512\text{ combinaisons}}
$$

**Limite :** RGB décrit les quantités de couleurs, mais ignore leur position dans l’image.

### Zernike : décrire la forme

Le seuillage d’**Otsu** sépare les pixels selon leur niveau de gris. Une règle fondée sur les bords aide à choisir le premier plan.

On obtient un **masque binaire** : premier plan à 1, fond à 0. Ce masque est recadré, redimensionné et centré.

Les moments de Zernike mesurent la contribution de différents motifs mathématiques à cette forme. Le code conserve leur **module**, c’est-à-dire leur grandeur. Cela réduit la sensibilité aux rotations, sans garantir une invariance parfaite sur des pixels.

Avec le degré maximal 8, le nombre de valeurs est :

$$
1+1+2+2+3+3+4+4+5=\boxed{25}
$$

**Limite :** si le masque représente mal l’objet, le descripteur décrit une mauvaise forme. Une image uniforme produit un vecteur nul.

### SFTA : analyser la texture

SFTA travaille sur une image en niveaux de gris. Avec **8 seuils**, il construit :

- 8 masques contenant les pixels au-dessus de chaque seuil ;
- 7 masques contenant les pixels entre deux seuils voisins.

Cela donne **15 masques**. Pour chaque masque, le programme extrait sa frontière et calcule :

1. une estimation de sa complexité par **box-counting** ;
2. la moyenne des niveaux de gris sur la frontière ;
3. le nombre de pixels de frontière.

Le box-counting consiste à couvrir la frontière avec des carrés de différentes tailles et à compter les carrés occupés. L’évolution de ce nombre permet d’estimer sa complexité.

La taille du descripteur est donc :

$$
(8+7)\times3=\boxed{45}
$$

**Limite :** ces valeurs dépendent des seuils, du cadrage et de la résolution. Le nombre de pixels de frontière n’est pas l’aire du masque.

## 4. Normalisation et fusion

Chaque famille est normalisée séparément avec la norme **L2** :

$$
\widehat f=\frac{f}{\sqrt{\sum_j f_j^2}}
$$

Par exemple, $[3,4]$ devient $[0{,}6,0{,}8]$. Un vecteur nul reste nul.

La normalisation réduit les différences d’amplitude entre familles. Elle intervient **après** l’extraction ; le redimensionnement intervient **avant**, sur les pixels.

La fusion assemble les blocs normalisés avec des poids, égaux à 1 par défaut :

$$
F=[w_c\widehat h_{\mathrm{RGB}},\ w_z\widehat z,\ w_t\widehat s]
$$

Avec les trois familles :

$$
512+25+45=\boxed{582}
$$

La normalisation d’un bloc ne garantit pas que toutes ses composantes aient la même influence, notamment dans SFTA.

## 5. Recherche et clustering

### Retrouver des images similaires

La stratégie par vecteurs classe les images par **distance euclidienne** croissante :

$$
d(I,J)=\sqrt{\sum_j\left(F_j(I)-F_j(J)\right)^2}
$$

Une petite distance indique une ressemblance selon les descripteurs. Elle ne garantit pas que les images représentent le même objet.

Une seconde stratégie, **RRF**, combine les classements obtenus séparément par les descripteurs. Son score n’est pas une distance.

### Regrouper les images

**K-Means** répartit les vecteurs en $K$ groupes autour de centres. Le programme compare plusieurs valeurs de $K$ et choisit automatiquement celle qui maximise la silhouette admissible sur la galerie.

La grille par défaut va de 2 à 10 lorsque la taille de la galerie le permet. Elle peut être modifiée. Un groupe appris ne correspond pas nécessairement à une catégorie réelle.

La recherche utilise toute la galerie : elle n’est pas limitée au cluster prédit.

## 6. Installation

Prérequis : **Python 3.12, 64 bits**.

~~~bash
git clone https://github.com/AHS0003/SimilarImage-Search-Clustering.git
cd SimilarImage-Search-Clustering
python -m venv .venv
~~~

Activer l’environnement sous Windows PowerShell :

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Sous Linux ou macOS :

~~~bash
source .venv/bin/activate
~~~

Installer les dépendances :

~~~bash
python -m pip install -r requirements-dev.txt
~~~

## 7. Utilisation

Placer les images dans des sous-dossiers correspondant aux classes, par exemple **dataset/chat/image001.jpg** et **dataset/chien/image001.jpg**.

Le dossier parent immédiat définit la classe. Il faut au moins trois images lisibles et uniques par classe.

Comparer les sept combinaisons de descripteurs et les deux stratégies de recherche :

~~~bash
python -m experiments.compare_descriptors --dataset dataset --output results/comparison --seed 42 --clusters auto --strategy both
~~~

Choisir un dossier de sortie neuf pour chaque passage. Les rapports existants ne sont pas écrasés.

Pour vérifier le fonctionnement avec des images synthétiques :

~~~bash
python -m experiments.generate_synthetic --output datasets/synthetic --seed 42 --per-class 10
python -m experiments.compare_descriptors --dataset datasets/synthetic --output results/synthetic-smoke --seed 42 --k-values 2 3 4 --dataset-kind synthetic
~~~

## 8. Évaluation et limites

Les fichiers identiques sont dédupliqués par SHA256 et les images illisibles sont filtrées avant la partition :

| Partie | Proportion cible | Utilisation |
|---|---:|---|
| Galerie | 60 % | Ajustement de K-Means, choix de K et index de recherche |
| Validation | 20 % | Requêtes évaluées séparément |
| Test | 20 % | Évaluation finale des réglages fixés |

La partition est reproductible et stratifiée lorsque les classes sont disponibles. Les arrondis peuvent modifier légèrement les proportions.

Les poids sont fixés avant l’évaluation. Le test ne doit pas servir à régler les paramètres.

| Mesure | Ce qu’elle évalue |
|---|---|
| Silhouette, Davies–Bouldin, Calinski–Harabasz | Cohésion et séparation des clusters de la galerie |
| ARI, NMI | Accord des clusters avec les classes connues |
| Precision@5, Precision@10 | Proportion de résultats pertinents aux premiers rangs |
| Recall@10 | Part des images pertinentes de la galerie retrouvées dans les dix premiers résultats |
| mAP@10 | Qualité du classement des résultats pertinents |

Une image est considérée comme pertinente si elle appartient à la même classe que la requête. Les requêtes sans image pertinente dans la galerie sont exclues des moyennes et comptabilisées dans la couverture. Une métrique indéfinie reste indisponible.

Les rapports CSV, les graphiques, le manifeste des partitions et les paramètres sont enregistrés dans le dossier de sortie.

**Points à vérifier sur les données réelles :**

- qualité des labels et des masques de premier plan ;
- effectifs et absence de chevauchement entre les partitions ;
- présence de quasi-doublons ou de réencodages, que SHA256 ne détecte pas ;
- cohérence des clusters et des résultats retournés.

Aucun meilleur descripteur ni score réel n’est revendiqué à ce stade.

## 9. Vérification du code

~~~bash
python -m pytest -q
python -m ruff check .
python -m pip check
~~~

Les formules complètes, les conventions d’implémentation et les références sont disponibles dans la [documentation technique](docs/technical-details.md).

# Crosstalk Estimation

Estimation du crosstalk polarimétrique sur la pile UAVSAR Rosamd_35012_04, en
reproduisant le protocole de Babu et al. (2022). Le projet fournit les
**baselines statistiques** (méthode de Quegan) et leur **validation aux
réflecteurs en coin** ; un encodeur à valeurs complexes (CVNN) leur sera ensuite
comparé.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Carte du projet : des `.slc` aux résultats

Le projet a **deux points d'entrée**, qui partagent le paquet `src/crosstalk/` :

- [src/run_baselines.py](src/run_baselines.py) applique Quegan patch par patch,
  sur la partition de test. C'est la **baseline** que le CVNN devra battre.
- [src/measure_reflectors.py](src/measure_reflectors.py) mesure la fuite aux
  réflecteurs en coin. C'est la **validation par vérité terrain**, comme chez
  Babu.

```
                    data/Rosamd_35012_04/  (20 .slc + 20 .ann + 1 .llh)
                         │                                   │
      ┌──────────────────┴──────────┐          ┌─────────────┴─────────────────┐
      │ CHAÎNE A  run_baselines.py  │          │ CHAÎNE B  measure_reflectors  │
      │ torchcvnn → cvnn → patchs   │          │ numpy.memmap direct, sans cvnn│
      └──────────────┬──────────────┘          └─────────────┬─────────────────┘
                     │        src/crosstalk/ (commun)        │
                     └──► covariance → quegan → physics ◄────┘
                                        │
               résiduel + réciprocité (par patch)    fuite aux réflecteurs (vérité terrain)
```

Les fonctions sont citées par leur nom plutôt que par leur numéro de ligne, qui
change à chaque modification du code.

### Les entrées

| fichier | contenu |
|---|---|
| `*_L090{HH,HV,VH,VV}_04_BC_s1_1x1.slc` | 5 dates × 4 polarisations = 20 fichiers de 61 349 lignes (azimut) × 9 874 colonnes (distance), en `complex64`, soit environ 4,8 Go chacun |
| `*.ann` | dimensions, cap de l'avion (−10,03°), `Polarimetrically Calibrated = Yes`, `Cross Talk Removed = No` |
| `*_2x8.llh` | latitude et longitude de chaque pixel de la grille 2×8 — chaîne B seulement |
| `data/corner_reflectors/rosamond_<date>.csv` | catalogue JPL des réflecteurs — chaîne B seulement |
| [configs/data_uavsar.yaml](configs/data_uavsar.yaml) | taille de patch, graine, ratios du découpage, normalisation, bloc `baselines:` |

**Notation.** Les `.slc` sont le **`s'` de Babu** (éq. 7 et 8), le `O` de Fore :
étalonnés en gain et en phase par JPL, mais **pas corrigés du crosstalk** — ce
que confirme la mesure du déséquilibre de canaux (voir « Le déséquilibre de
canaux »). Le modèle s'écrit `s' = D s + n`, où `s` est le signal vrai : la
chaîne estime `D` puis corrige, `s_chapeau = Sigma s'`. La distorsion n'est
jamais appliquée aux données réelles.

### Chaîne A — `python src/run_baselines.py`

| # | étape | code | entrée → sortie |
|---|---|---|---|
| A1 | lire la configuration | `run_baselines.main` | YAML → `dict` |
| A2 | instancier les méthodes | `run_baselines.build_estimators` → `estimators/registry.create_from_config` | `{name: quegan, window: [5, 5]}` → `QueganEstimator(window=(5, 5))`, que le décorateur `@register_estimator` a inscrit sous le nom `quegan` |
| A3 | fixer le hasard, compléter la configuration | `run_baselines.build_dataloader` → `cvnn.utils.set_seed(38)`, puis `cvnn/data.validate_and_correct_config` | ajoute `type: polsar`, `has_labels: False`, `num_channels: 4`, lus dans `cvnn/dataset_registry.py` |
| A4 | découvrir les fichiers | `cvnn/data._create_dataset` → `torchcvnn.SLCDataset` | groupe les `.slc` par date, lit les `.ann` ; grille de 239 × 38 = **9 082 patchs de 256 px par date**, 45 410 au total. Aucun pixel n'est encore lu |
| A5 | **découper train / valid / test** | `cvnn/data._groupes_par_emplacement`, appelée par `get_dataloaders` | mélange les 9 082 emplacements au sol, les 5 dates suivent leur emplacement → 36 330 / 4 540 / 4 540 patchs, sans fuite |
| A6 | construire les transformations | `cvnn/transform_registry.build_transform_pipeline` | empilement du `dict` en ordre alphabétique (HH, HV, VH, VV) = `POL_ORDER`, tenseur, puis `GlobalScalarNormalize` qui divise par 0,16444 = √(2 × 0,013521) |
| A7 | construire les chargeurs | `cvnn/data.get_dataloaders` | seul `test_loader` sert : `shuffle=False`, lots de 16, soit 284 lots |
| A8 | lire les patchs | `run_baselines.iter_patches` → `positions_de_test` → `position_du_patch` | lot `(16, 4, 256, 256)` `complex64` → un couple `(position, patch)` à la fois : la position — acquisition, ligne et colonne du coin haut-gauche — se déduit de `Subset.indices`, de `SLCDataset.patch_counts` et de la grille de `SLCFile` ; un chargeur mélangé est refusé |
| A9 | boucle d'évaluation | `run_baselines.evaluate_patches` | une ligne par couple patch/méthode, qui commence par la position du patch ; une `CrosstalkError` donne le statut `ill_conditioned` sans arrêter la boucle |
| A10 | estimer, corriger, ré-estimer | `metrics.residual_crosstalk` | détaillé en A10a-e ci-dessous, et tracé sur un patch réel dans « Trace sur un patch de contrôle » |
| A10a | covariance | `covariance.polarimetric_covariance` | 16 produits `S'_i S'_j*`, table de sommes cumulées → `C` de forme `(252, 252, 4, 4)` `complex128`, soit **63 504 fenêtres 5×5** |
| A10b | formes fermées | `QueganEstimator._solve` | par fenêtre : `Δ`, `u, v, w, z` (Babu, équation 13), `X`, `alpha` ; masque `valides` : rapports finis et au-dessus des seuils |
| A10c | moyenne | `QueganEstimator.estimate` | moyenne des `theta` sur les fenêtres valides → un `CrosstalkParameters` par patch |
| A10d | correction | `CrosstalkEstimator.correct` → `physics.inverse_distortion_matrix` → `physics.apply_matrix` | `Sigma` `(4, 4)`, protégée contre `(uw−1)(vz−1) ≈ 0`, puis `s_chapeau = Sigma · s'` pour chaque pixel |
| A10e | ré-estimation et réciprocité | `estimate(s_chapeau)`, `metrics.reciprocity` | `ResidualReport` : initial, résiduel, réciprocité avant et après |
| A11 | synthèse et sauvegarde | `run_baselines.ecrire_resultats`, `run_baselines.summarize` | les 4 540 lignes → `experiments/baselines.csv`, réécrit à chaque lancement ; leur médiane → tableau affiché |

### Chaîne B — `python src/measure_reflectors.py [--desequilibre]`

Ni cvnn ni normalisation : les `.slc` sont ouverts en `numpy.memmap` et lus aux
seuls endroits utiles.

| # | étape | code | ce qui se passe |
|---|---|---|---|
| B1 | pour chaque date | `main` → `mesure_une_acquisition` | ouvre 4 memmaps `(61 349, 9 874)` et le `.llh` |
| B2 | **localiser les réflecteurs** | `localise_les_reflecteurs` | lit le catalogue (`lire_catalogue`) ; garde les réflecteurs orientés vers le radar (`oriente_vers_le_radar`, ±45° du cap) et les 8 de Babu (`est_retenu_par_babu`) ; les place par plus proche voisin dans le `.llh`, ×8 en azimut et ×2 en distance (`position_radar`) ; cherche le pic de `\|HH\|` à ±60 px (`affine_sur_le_pic`) ; contrôle la signature de trièdre (`est_un_triedre`) |
| B3 | déséquilibre de canaux, avec `--desequilibre` seulement | `desequilibre_de_l_acquisition` → `crosstalk/imbalance.py` | `f` et `phi_t + phi_r` aux 8 pics ; `g` et `phi_t - phi_r` sur toute l'image, par tranches de 2 048 lignes → matrice diagonale `K` |
| B4 | **`theta` de la bande** | `theta_de_la_bande` | colonnes `c ± 10` sur les 61 349 lignes, en tranches de 8 192 (`K` appliquée si présente) ; Quegan 5×5 par tranche, puis moyenne des `theta`. C'est ce qui rend la mesure non circulaire |
| B5 | mesurer | `mesure_un_reflecteur` → `metrics.corner_reflector_crosstalk` | « avant » = `20 log10(max(\|HV\|, \|VH\|) / sqrt(\|HH\| \|VV\|))` au pic brut ; « après » = la même mesure sur `Sigma · s'`, où `s'` est le pic brut — ou `K ·` pic brut avec `--desequilibre` |
| B6 | agréger | `agrege`, `resume` | tableau par date, moyenne (Babu) et médiane, puis l'ensemble des 40 mesures |

### Ordre de lecture recommandé

Des fondations vers les points d'entrée : chaque fichier ne dépend que de ceux
qui le précèdent.

| # | fichier | à vérifier en le lisant |
|---|---|---|
| 1 | [configs/data_uavsar.yaml](configs/data_uavsar.yaml) | `seed`, `patch_size`, ratios, `window: [5, 5]` |
| 2 | [cvnn/src/cvnn/data.py](cvnn/src/cvnn/data.py) — `_create_dataset`, `_groupes_par_emplacement`, `get_dataloaders` | le découpage groupé par emplacement ; le test chargé sans mélange |
| 3 | [cvnn/src/cvnn/transform_registry.py](cvnn/src/cvnn/transform_registry.py) — `GlobalScalarNormalize` | la normalisation est **un seul scalaire réel**, donc `theta` invariant |
| 4 | [src/crosstalk/exceptions.py](src/crosstalk/exceptions.py) | les deux cas d'échec : D singulière, covariance mal conditionnée |
| 5 | [src/crosstalk/parameters.py](src/crosstalk/parameters.py) | ordre `(u, v, w, z, alpha)`, conversions en dB |
| 6 | [src/crosstalk/physics.py](src/crosstalk/physics.py) | `POL_ORDER`, D réindexée de Fore, `Sigma · D = I`, `apply_matrix` |
| 7 | [src/crosstalk/covariance.py](src/crosstalk/covariance.py) | table de sommes cumulées, forme `(H − 4, W − 4, 4, 4)` |
| 8 | [src/crosstalk/estimators/base.py](src/crosstalk/estimators/base.py), puis [registry.py](src/crosstalk/estimators/registry.py) | interface abstraite, enregistrement par décorateur |
| 9 | [src/crosstalk/estimators/quegan.py](src/crosstalk/estimators/quegan.py) | formes fermées, seuils relatifs, `isfinite`, moyenne des `theta` |
| 10 | [src/crosstalk/metrics.py](src/crosstalk/metrics.py) | le résiduel est **circulaire**, la réciprocité ne l'est pas ; formule au réflecteur |
| 11 | [src/run_baselines.py](src/run_baselines.py) | assemblage de la chaîne A |
| 12 | [src/crosstalk/imbalance.py](src/crosstalk/imbalance.py) | modèle de Babu, équation 7 ; ambiguïté de `pi` sur les phases |
| 13 | [src/measure_reflectors.py](src/measure_reflectors.py) | assemblage de la chaîne B ; la bande comme rempart contre la circularité |
| 14 | [src/crosstalk/synthetic.py](src/crosstalk/synthetic.py) et [tests/](tests/) | hors chaîne : ne servent qu'aux tests, à `theta` connu |

**Quatre points à garder en tête en relisant.**

- **L'ordre des canaux est implicite** : il naît en A6, de l'ordre alphabétique
  du `dict` rendu par torchcvnn, et tout le reste en dépend. Un test verrouille
  `POL_ORDER == sorted(POL_ORDER)`.
- **Les acquisitions ne sont pas dans l'ordre chronologique** : `SLCDataset` les range
  dans l'ordre de `glob` — 160610, 160211, 160920, 160520, 160816 sur cette machine.
  `experiments/baselines.csv` les désigne donc par leur clé, jamais par un rang.
- **Les deux chaînes n'ont pas la même normalisation**, sans effet : `theta` et la
  fuite au réflecteur sont des rapports.
- **La fenêtre 5×5 est fixée en dur dans la chaîne B** (`main` de
  `measure_reflectors.py`), et non lue dans le YAML : `full` y rendrait la mesure
  circulaire — voir « Le piège de circularité » plus bas.

## Pipeline de données

Assuré par [`torchcvnn`](https://github.com/torchcvnn/torchcvnn) (lecture des
`.slc`, grille de patchs) et [`cvnn`](https://github.com/QuentinGABOT/cvnn)
(découpage, transformations, `DataLoader`) ; leur enchaînement est détaillé aux
étapes A3 à A8 de la carte du projet. Le seul code propre au projet est le
branchement de `UavsarSLC` dans `cvnn`, décrit ci-dessous. Les données attendues
sont dans `data/Rosamd_35012_04/`, soit environ 97 Go pour les 20 `.slc`.

Pour charger les patchs, par exemple côté CVNN :

```python
import sys, yaml
sys.path.insert(0, "cvnn/src")
from cvnn.data import get_dataloaders, validate_and_correct_config

cfg = validate_and_correct_config(yaml.safe_load(open("configs/data_uavsar.yaml")))
train_loader, valid_loader, test_loader = get_dataloaders(cfg, use_cuda=True)

batch = next(iter(train_loader))    # (B, 4, H, W) complex64, canaux [HH, HV, VH, VV]
```

### Branchement dans cvnn

`UavsarSLC` a été ajouté au framework comme n'importe quel autre dataset :

- `cvnn/src/cvnn/dataset_registry.py` — entrée `UavsarSLC`
  (`type: polsar`, 4 canaux, sans étiquettes) ;
- `cvnn/src/cvnn/data.py` — import de `SLCDataset`, branche
  `elif dataset_name == "UavsarSLC"` dans `_create_dataset`, et `UavsarSLC`
  ajouté à la liste des datasets à `patch_size` ;
- `cvnn/src/cvnn/data.py` — `_groupes_par_emplacement` : sur une pile
  co-enregistrée, le découpage mélange les **emplacements au sol** et non les
  patchs, pour que les 5 dates d'un même lieu tombent du même côté. Le mélange
  patch par patch plaçait 99,9 % des emplacements de test aussi dans le train.

`cvnn/` est versionné dans ce dépôt ; ces modifications restent à reporter en
amont dans le dépôt `cvnn`.

### Point d'attention : la normalisation

Les deux normalisations de `cvnn` ramènent **chaque canal** à variance unité :
`normalize` blanchit chaque polarisation par une matrice 2×2,
`global_scalar_normalize` lui applique un scalaire — par canal lui aussi.
Mesuré sur cette pile :

| | HH | HV | VH | VV |
|---|---|---|---|---|
| brut | 1,00 | 0,185 | 0,213 | 1,29 |
| `normalize` | 1,00 | 0,553 | 0,563 | 1,68 |
| `global_scalar_normalize` (cov. par canal) | 1,00 | 0,592 | 0,628 | 1,79 |

Les canaux croisés remontent d'un facteur ~3. Ce n'est pas une perte
d'information — une mise à l'échelle par canal est une matrice diagonale Λ fixe
et inversible — mais elle change ce que le réseau estime : le modèle mesuré
`s' = D·s` devient `Λ·s' = (Λ·D)·s`, et les paramètres de crosstalk,
qui sont des **rapports** de coefficients de `D`, sont multipliés par σⱼ/σᵢ. Il
faut alors appliquer Λ⁻¹ pour revenir à la grandeur physique.

Sous une homothétie **globale** `α`, au contraire, `D` devient `α·D` et ces
rapports sont invariants : la sortie du réseau est directement la grandeur
physique. La configuration livrée obtient cela **sans code**, en donnant la
*même* covariance aux 4 canaux (`cov_complex_value`). L'échelle est standardisée
(module moyen ~0,38) et les rapports entre polarisations restent exacts.

Si vous remettez `recompute_statistics: True`, `cvnn` réécrit ces valeurs avec
une covariance par canal et le comportement du tableau ci-dessus revient.

C'est aussi ce qui laisse les **baselines** intactes : `GlobalScalarNormalize`
divise ici les 4 canaux par le même scalaire (0,16444), si bien que `theta`, la
réciprocité et la fuite aux réflecteurs, qui sont des rapports, n'en dépendent
pas.

## Estimation du crosstalk

Paquet `src/crosstalk/`, indépendant du réseau : il fournit les **baselines
statistiques** contre lesquelles le CVNN sera comparé. Seule la méthode de
Quegan est implémentée à ce jour.

```bash
python src/run_baselines.py                         # configuration par défaut
python src/run_baselines.py --config autre.yaml     # ou une autre configuration
```

```
         méthode  patchs  échecs  initial dB  résiduel dB  réduction dB  |alpha|  récip. avant  récip. après
------------------------------------------------------------------------------------------------------------
          quegan    4540       0       -21.1        -30.7           9.6    1.226         0.821         0.846
```

Les deux dernières colonnes sont la validation indépendante décrite plus bas :
la réduction de 9,6 dB est jugée par l'estimateur lui-même, la réciprocité ne
l'est pas.

Chaque colonne est la **médiane** sur les patchs, un `theta` étant estimé par
patch. Ces valeurs dépendent de `seed:`, qui fixe le découpage train / valid /
test : changer la graine change les patchs évalués, pas la méthode. Sortie
ci-dessus : `seed: 38`, découpage groupé par emplacement au sol. Durée :
environ 6 min 30 s.

Le détail est écrit à chaque lancement dans `experiments/baselines.csv`, en
écrasant le précédent ; le dossier est ignoré par git. Une ligne par patch et par
méthode :

| colonnes | contenu |
|---|---|
| `acquisition`, `ligne`, `colonne` | clé torchcvnn de l'acquisition, coin haut-gauche du patch en pixels 1×1 (azimut, distance) |
| `name`, `status` | la méthode, `ok` ou `ill_conditioned` |
| `initial_db` … `reciprocity_after` | les six grandeurs du tableau ci-dessus |
| `u_re`, `u_im`, …, `alpha_re`, `alpha_im` | le `theta` initial, en parties réelle et imaginaire |

La position sert de clé pour comparer deux méthodes patch à patch, sur les mêmes
patchs. Le `theta` complet sert à la cohérence en azimut : le crosstalk ne
dépendant que de la distance, les `theta` d'une même colonne et d'une même date
ne devraient différer que par l'erreur d'estimation — `alpha` compris, que ni la
réciprocité ni les réflecteurs ne voient. Une ligne `ill_conditioned` laisse ses
cellules de mesure vides.

La fenêtre glissante 5×5 du protocole de Babu coûte environ 12× le mode `full`
en temps de calcul. Tout se règle dans le bloc `baselines:` de
[configs/data_uavsar.yaml](configs/data_uavsar.yaml). Le script se lance depuis
n'importe quel dossier : un `--config` relatif se résout depuis le dossier courant,
ou à défaut depuis la racine du dépôt, et le `trainpath` du YAML toujours depuis
la racine. Le modèle physique et la
covariance sont **communs** à toutes les méthodes : une méthode nouvelle
n'écrit que `estimate`, et s'inscrit au registre par `@register_estimator`.

### Trace sur un patch de contrôle

Les étapes A10a à A10e de la carte du projet, sur le patch `(30000, 4500)` de
l'acquisition du 11 février :

```
ENTRÉE   HH 0.007157   HV 0.000731   VH 0.001004   VV 0.007269   (puissances moyennes)
A10a     C de forme (252, 252, 4, 4) : 63 504 fenêtres 5×5, une matrice chacune
         fenêtres retenues : 63 504 / 63 504
A10b-c   moyenne des 63 504 theta
         u =  0.024319 + 0.005222j    fuite HH -> VH
         z =  0.046798 - 0.025522j    fuite HH -> HV
         v = -0.072366 - 0.033255j    fuite VV -> VH
         w = -0.013079 - 0.015735j    fuite VV -> HV
         |alpha| = 1.290804, arg(alpha) = -1.30°   déséquilibre HV / VH
         crosstalk = -21.98 dB
A10d-e   résiduel = -32.59 dB         réduction 10.62 dB
```

**Le crosstalk se lit dans la covariance, pas dans les pixels.** L'étape A10a réduit
les 25 pixels × 4 canaux de chaque fenêtre à une matrice 4×4. Toute
l'information y est : Quegan suppose que sur le signal **vrai**, les termes co-pol/cross-pol
`<HH·VH*>` et `<VV·HV*>` sont **nuls** par symétrie de réflexion (équation 17 de
Fore). Tout ce qu'on mesure à ces positions sur le signal **observé** `s'` est donc dû
au crosstalk — et c'est ce que lisent les formes fermées, mais au premier ordre
et en négligeant la puissance croisée : voir « Le biais propre de Quegan ».

**`apply_distortion` ne tourne jamais sur les données réelles.** Elle sert
uniquement aux tests, pour fabriquer une observation à `theta` connu — le seul
moyen de disposer d'une vérité terrain, puisque les `.slc` n'en ont pas. C'est
ainsi qu'est mesuré le biais propre de la méthode : 1 % sans bruit et à un
million d'échantillons quand la puissance croisée est négligeable
(`beta = 0,002`), mais de l'ordre de 30 % au `beta` médian de cette pile.

### Le biais propre de Quegan

Les formes fermées ne sont exactes qu'à puissance croisée nulle. La
démonstration tient en quatre étapes ; `O` désigne le signal observé `s'`, `S`
le signal vrai `s`.

**1. Le modèle, ligne par ligne.** Les quatre lignes de `D`
(`physics.distortion_matrix`) donnent, pour chaque pixel :

```
O_HH = S_HH      + (v/√α)·S_HV  + w√α·S_VH  + vw·S_VV
O_HV = z·S_HH    + (1/√α)·S_HV  + wz√α·S_VH + w·S_VV
O_VH = u·S_HH    + (uv/√α)·S_HV + √α·S_VH   + v·S_VV
O_VV = uz·S_HH   + (u/√α)·S_HV  + z√α·S_VH  + S_VV
```

Quegan fait deux hypothèses sur le signal vrai (Fore, équation 17 ;
`synthetic.quegan_covariance`) :

```
symétrie de réflexion :  <S_HH·S_X*> = <S_VV·S_X*> = 0
réciprocité           :  S_HV = S_VH = S_X,  de puissance P_X = <|S_X|²>
```

La corrélation co-polaire `<S_HH·S_VV*>` reste libre.

**2. Développer une corrélation.** `C31 = <O_VH·O_HH*>` est le produit de deux
sommes de quatre termes, soit seize produits. Ceux qui associent un canal
co-polaire à `S_X` sont nuls par symétrie de réflexion. Au premier ordre en
crosstalk, il reste :

| produit | valeur moyenne | ordre |
|---|---|---|
| `u·S_HH · S_HH*` | `u·σ_HH` | 1 |
| `v·S_VV · S_HH*` | `v·<S_VV·S_HH*>` | 1 |
| `√α·S_X · (v/√α·S_X)*` | `e^{i arg α}·v*·P_X` | 1 |
| `√α·S_X · (w√α·S_X)*` | `\|α\|·w*·P_X` | 1 |
| tous les autres | — | 3 |

avec `√α / √α* = e^{i arg α}` et `√α · √α* = |α|` pour la racine principale,
celle de `np.sqrt`. Remplacer ensuite `σ_HH` par `C11` et `<S_VV·S_HH*>` par
`C41`, qui n'en diffèrent que de `O(θ²)`, coûte un terme en `θ·θ² = O(θ³)`. Le
même calcul sur les trois autres corrélations co-pol/cross-pol donne :

```
C31 = u·C11 + v·C41 + e₁ + O(θ³)        e₁ = P_X·( e^{i arg α}·v* + |α|·w* )
C34 = u·C14 + v·C44 + e₂ + O(θ³)        e₂ = P_X·( e^{i arg α}·u* + |α|·z* )
C21 = z·C11 + w·C41 + e₃ + O(θ³)        e₃ = P_X·( v*/|α| + e^{−i arg α}·w* )
C24 = z·C14 + w·C44 + e₄ + O(θ³)        e₄ = P_X·( u*/|α| + e^{−i arg α}·z* )
```

Physiquement, le signal croisé vrai `S_X` est présent dans VH, et il fuit aussi
dans HH par `v` et `w` : HH et VH partagent une composante sans rapport avec
`u`. C'est le terme `e₁`.

**3. Ce que résout Quegan.** Supprimer `e₁` et `e₂` laisse un système 2×2 :

```
| C11  C41 | | u |   | C31 |
| C14  C44 | | v | = | C34 |          Δ = C11·C44 − |C14|²
```

dont la solution par la règle de Cramer est :

```
u = (C44·C31 − C41·C34) / Δ          v = (C11·C34 − C14·C31) / Δ
```

C'est mot pour mot l'équation 13 de Babu et la ligne de
`QueganEstimator._solve` ; `(z, w)` s'obtiennent de même à partir de `C21` et
`C24`. Les formes fermées sont donc la solution **exacte** du système privé des
termes en `P_X`, et non une approximation de plus.

**4. Le biais.** Les corrélations observées contiennent `e₁` à `e₄`. Injectées
dans les formes fermées, elles donnent, au premier ordre :

```
δu = û − u = (C44·e₁ − C41·e₂) / Δ          δv = v̂ − v = (C11·e₂ − C14·e₁) / Δ
δz = ẑ − z = (C44·e₃ − C41·e₄) / Δ          δw = ŵ − w = (C11·e₄ − C14·e₃) / Δ
```

C'est ce que vérifie
`test_le_biais_est_le_terme_croise_neglige_par_les_formes_fermees`, à 1 % près,
sur la covariance théorique. L'ordre de grandeur s'en déduit, avec
`ρ_HHVV = C14 / √(C11·C44)` :

```
e₁, e₂ ∝ P_X·θ ;   numérateur ∝ σ_co·P_X·θ ;   Δ = C11·C44·(1 − |ρ_HHVV|²) ∝ σ_co²
⟹  δu ∝ (P_X / σ_co) · θ / (1 − |ρ_HHVV|²)
```

Le biais **relatif** `δu / θ` est donc proportionnel au rapport
`beta = P_X / σ_co`, et indépendant du niveau de crosstalk
(`test_le_biais_relatif_est_proportionnel_au_rapport_crosspol_copol`,
`test_le_biais_relatif_ne_depend_pas_du_niveau_de_crosstalk`). C'est d'ailleurs
la seule combinaison possible : `u` est sans dimension, et l'estimateur est
invariant à une homothétie globale (`test_invariance_a_une_homothetie_reelle_globale`).

Le coefficient de proportionnalité dépend en revanche de la direction de
`theta`, de `alpha` et de la corrélation HH–VV, par le facteur
`1 / (1 − |ρ_HHVV|²)` et par `C41` au numérateur. Mesuré comme le plus grand
biais rapporté au plus grand terme de crosstalk, il va de 0,6 à 3,9 fois `beta`
selon les configurations essayées, et vaut 2,3 pour le `theta` du patch de
contrôle avec `|rho_HHVV| = 0,5`. Le reste en `O(θ³)`, soit `O(|θ|²)` en
relatif, ne domine qu'à `beta` très faible — d'où le 1 % ci-dessus.

**Sur la pile.** `beta`, mesuré sur 300 patchs 256×256 tirés au hasard (60 par
date), vaut 0,134 en médiane — quartiles 0,081 et 0,184, 9e décile 0,219. C'est
une borne haute : le bruit thermique gonfle la puissance croisée mesurée sans
entrer dans le biais, puisqu'il n'est pas corrélé entre HH et VH. Biais exact sur
la covariance théorique, avec le `theta` du patch de contrôle (−21,98 dB) :

| `beta` | `\|rho_HHVV\| = 0,5` | toutes les corrélations HH–VV essayées |
|---|---|---|
| 0,081 (1er quartile) | −36,5 dB | −42,3 à −32,0 dB |
| 0,134 (médiane) | **−32,2 dB**, 31 % du crosstalk | −37,9 à −27,6 dB |
| 0,184 (3e quartile) | −29,4 dB | −35,1 à −24,9 dB |

Les corrélations essayées vont de 0,3 à 0,7 en module, sous quatre phases ; la
médiane de `|rho_HHVV|` sur la pile est 0,509. Le biais atteint donc le niveau
de l'exigence système de −30 dB que cite Babu (§4.1).

Le résiduel en porte la trace. Sur les mêmes 300 patchs, en fenêtre 5×5 :

| `beta` du patch | patchs | résiduel médian |
|---|---|---|
| < 0,05 | 28 | −32,5 dB |
| 0,05 – 0,1 | 79 | −32,3 dB |
| 0,1 – 0,2 | 149 | −29,5 dB |
| > 0,2 | 44 | −26,6 dB |

Il croît avec `beta` (`rho_s = +0,53`), plus nettement encore avec
`beta × |theta|` (`rho_s = +0,71`), et sa médiane de −30,3 dB tombe dans la plage
prédite.

Ce biais n'est pas une erreur d'implémentation : c'est l'approximation de la
méthode, que partagent Babu (§3.3.1) et Fore. Le système d'Ainsworth (Babu,
équations 25 à 27), non implémenté ici, contient au contraire ces termes : sa
matrice `tau` porte les puissances croisées `C22`, `C23`, `C32`, `C33`, et la
séparation en parties réelle et imaginaire y traite les conjugués.

### Convention d'ordre des canaux

**C'est le point le plus délicat du projet.** Une erreur d'ordre ne lève aucune
exception : elle renomme silencieusement les paramètres estimés.

L'ordre retenu est celui que sert le pipeline, `POL_ORDER = [HH, HV, VH, VV]`,
parce que `SLCDataset` trie les polarisations alphabétiquement. Aucune
permutation n'existe donc nulle part dans le paquet, et l'invariant qui le
justifie — `POL_ORDER == sorted(POL_ORDER)` — est verrouillé par un test.

Fore et al. (2015) ordonnent en revanche `[HH, VH, HV, VV]`. Leurs équations
sont donc **réindexées** dans le code : la formule est la même, mais les chiffres
1-2-3-4 qu'elle contient ne désignent pas les mêmes canaux.

| indice | chez Fore | ici |
|---|---|---|
| 1 | HH | HH |
| **2** | **VH** | **HV** |
| **3** | **HV** | **VH** |
| 4 | VV | VV |

Les indices 1 et 4 ne bougent jamais : toute expression dont les deux indices
sont dans `{1, 4}` — `Delta = C11 C44 - |C14|²`, par exemple — se recopie telle
quelle.

Réindexer ne change **aucun** résultat : sur un patch réel, notre estimateur et
les équations 24-26 de Fore appliquées dans son propre ordre rendent
`u = 0.053956 - 0.038570j` de part et d'autre, à `6.9e-18` près. Ce qui change
les résultats, c'est de ne **pas** réindexer : les `(u, v, w, z, alpha)`
ressortent alors comme `(z, w, v, u, 1/alpha)`, sans la moindre erreur levée.

### Les deux articles de référence

- **Fore et al. (2015)**, *UAVSAR Polarimetric Calibration*, IEEE TGRS 53(6).
  Équations 11 à 15 (modèle), 16-17 (covariance), 24 à 26 (formes fermées de
  Quegan).
- **Babu et al. (2022)**, *Polarimetric Calibration and Spatio-Temporal
  Polarimetric Distortion Analysis of UAVSAR PolSAR Data*, Earth and Space
  Science 9. Équations 8 à 18.

**Règle suivie** : le protocole et les équations de Babu, sauf là où elles sont
incohérentes ou incomplètes, auquel cas celles de Fore. Fore est en effet le seul
des deux articles qui forme un système fermé : `Sigma_eq15 @ D_eq13 = I` à
`2.2e-16`, et ses équations 24-26 retrouvent exactement le `theta` injecté par
son équation 13. C'est donc sur lui que les tests ancrent `D` et `Sigma`.

Babu adopte l'ordre de canaux retenu ici, mais ne réindexe qu'une partie de ses
équations, ce qui rend l'article internement incohérent :

| équation de Babu | état | conséquence |
|---|---|---|
| éq. 8 (`D` en produit) | = éq. 13 de Fore à l'identique, **non** réindexée | contredit son éq. 13 |
| éq. 10 (`D` développée) | non réindexée **et** fautive : colonne HV divisée par `sqrt(alpha)` au lieu d'être multipliée (facteur `1/alpha` mesuré sur 3 entrées) | inutilisable |
| éq. 12 (`Sigma`) | même faute de transcription | n'inverse aucune de ses deux `D` (`2.1e-02`) |
| **éq. 13** (formes fermées) | **correctement réindexée** | la seule utilisable telle quelle |
| éq. 16-18 (`alpha`) | = éq. 26 de Fore à l'identique, non réindexée | contredit son éq. 13 |

Chaque article contient cependant une erreur que l'autre corrige : le terme sous
la racine de `alpha`. Sur un `alpha` vrai de `1.3000` :

| forme | source | résultat |
|---|---|---|
| `4\|alpha_2\|²` | Babu éq. 15 | **1.2990** ✓ |
| `4 + \|alpha_2\|²` | texte de Fore éq. 25 | 1.2199 ✗ |

Le code retient la forme de Babu, verrouillée par l'identité
`x - 1/x = |alpha_1| - 1/|alpha_2|`.

#### Correspondance équation ↔ code

| Fore, ordre `[HH, VH, HV, VV]` | code, ordre `[HH, HV, VH, VV]` |
|---|---|
| éq. 16 `C_ij = <O_i O_j*>` | identique, notée `C_ij = <S'_i S'_j*>` |
| éq. 17 `C_Q` (covariance vraie) | **identique** — le bloc `beta` est symétrique en 2↔3 |
| éq. 13 `D`, éq. 15 `Sigma` | lignes **et** colonnes 2↔3 |
| éq. 24 `Delta = C11 C44 - \|C14\|²` | identique |
| éq. 24 `u = (C44 C21 - C41 C24)/Delta` | `u = (C44 C31 - C41 C34)/Delta` |
| éq. 24 `v = (C11 C24 - C21 C14)/Delta` | `v = (C11 C34 - C31 C14)/Delta` |
| éq. 24 `z = (C44 C31 - C41 C34)/Delta` | `z = (C44 C21 - C41 C24)/Delta` |
| éq. 24 `w = (C11 C34 - C31 C14)/Delta` | `w = (C11 C24 - C21 C14)/Delta` |
| éq. 26 `X = C32 - z C12 - w C42` | `X = C23 - z C13 - w C43` |
| éq. 26 `alpha_1 = (C22 - u C12 - v C42)/X` | `alpha_1 = (C33 - u C13 - v C43)/X` |
| éq. 26 `alpha_2 = X*/(C33 - z* C31 - w* C34)` | `alpha_2 = X*/(C22 - z* C21 - w* C24)` |

Un écart délibéré à son équation 25 : le `4|alpha_2|²` ci-dessus. Son facteur
de phase `alpha_1/|alpha_1|` est en revanche **appliqué**, alors que l'équation 15
de Babu l'omet et rend un `alpha` réel. Sans lui, la correction laisserait un
déphasage `arg(alpha)` entre HV et VH — jusqu'à environ −20° le 20 mai, d'après
le `phi_t - phi_r` mesuré. Ce choix ne change aucun chiffre comparable à Babu :
la phase n'agit que par `diag(1, e^{-i phi/2}, e^{+i phi/2}, 1)`, et la fuite aux
réflecteurs, la réciprocité et le résiduel ne lisent que des modules
(`test_la_phase_de_alpha_ne_change_aucune_metrique_comparable_a_babu`).

En fenêtre glissante, le module et la phase de `alpha` se moyennent séparément :
moyenne des modules, comme chez Babu dont le `alpha` est réel, et phase de la
moyenne complexe. La phase varie d'environ 23° d'une fenêtre 5×5 à l'autre, si
bien que moyenner les complexes ferait s'annuler une partie du module : sur 40
patchs, −5,3 % contre l'estimation plein patch, au lieu de +3,5 % pour la moyenne
des modules.

⚠️ En comparant le code au papier, ne « corrigez » pas `C31` en `C21` : c'est la
réindexation, et `tests/test_physics.py::test_d_est_la_matrice_de_fore_reindexee`
la verrouille.

### Ce qui n'est pas implémenté

Le paquet suit **Babu et al. (2022), §3.3.1** — le Quegan de base. Quatre écarts
subsistent, chacun assumé pour une raison précise.

**Le moyennage par ligne de distance.** Babu moyenne les `theta` le long d'une
ligne de distance puis calibre cette ligne. Le projet moyenne sur tout le patch
et rend **un `theta` par patch**, parce que le CVNN à venir produira lui aussi un
jeu par patch : une baseline à 252 jeux par patch ne lui serait pas comparable.
`QueganEstimator.estimate_range_lines` implémente le protocole de Babu et reste
disponible ; l'écart mesuré entre les deux est de **0,79 dB** sur le crosstalk.

**Le masque de corrélation — délibérément absent.** Le Quegan de Babu §3.3.1
n'en a aucun. Le masque est ce qui définit l'**improved Quegan** du §3.3.2, qui
écarte les pixels dont `|r(HH,VH)|` ou `|r(VV,HV)|` dépasse 0,4 (Kimura et al.,
2004, équation 19) et modifie en outre l'estimation de `alpha` (son équation 20,
avec le rapport de bruit `m`). L'ajouter ici ferait basculer l'estimateur hors de
son périmètre sans changer son nom.

⚠️ Ne pas confondre avec le masque de **Fore §V-B**, qui porte sur un autre
couple — `|r(HH,VV)| > 0,2` — et n'est pas le même critère. Mesuré sur 250 patchs
de cette pile : celui de Kimura en conserve **91 %**, celui de Fore **21 %**, et
les deux sont statistiquement décorrélés (`rho_s = -0,10`, non significatif).
Fore justifie pourtant le sien en invoquant la nullité des corrélations
co-pol/cross-pol — or son équation 17 laisse le terme HH–VV explicitement
**non nul**. Sa justification décrit les couples de Kimura, pas le sien.

**Ainsworth et al. (2006)**, méthode itérative décrite en annexe A-A de Fore et
au §3.3.3 de Babu. Non implémentée.

**L'étalonnage radiométrique absolu (Babu §3.1).** Le gain `A` n'est pas estimé :
le crosstalk aux réflecteurs est un rapport de canaux et les seuils de Quegan sont
relatifs, si bien qu'un gain global s'annule partout. L'estimer demanderait la
SER théorique de chaque réflecteur (son équation 2) pour un effet nul. Le
déséquilibre de canaux de son §3.2, lui, est implémenté (`imbalance.py`) et
disponible par `measure_reflectors.py --desequilibre`.

### Validation : trois niveaux de preuve

Les `.slc` n'ont pas de vérité terrain : on ne peut pas savoir si un `theta`
estimé est juste. Le projet s'appuie donc sur trois contrôles d'exigence
croissante, dont deux sont indépendants de l'estimateur.

| niveau | ce qui juge | où |
|---|---|---|
| 1. crosstalk résiduel | **l'estimateur lui-même** | `run_baselines.py` |
| 2. réciprocité HV/VH | la physique du capteur | `run_baselines.py` |
| 3. réflecteurs en coin | la réponse connue d'un trièdre | `measure_reflectors.py` |

#### Niveau 1 — le résiduel, et pourquoi il ne prouve rien

`residual_crosstalk` estime, corrige, puis **ré-estime avec le même estimateur**.
La réduction obtenue est donc optimiste par construction, et dominée par le
plancher propre de la méthode plutôt que par ce que la correction a retiré.

Deux mesures le montrent. Les modes `full` et `(5, 5)` diffèrent de 0,9 dB sur
le crosstalk initial mais s'accordent à 0,1 dB sur le résiduel : les deux butent
sur le même plancher. Ce plancher s'explique par le biais propre de Quegan, qui croît avec
le rapport `beta` des puissances croisée et co-polaire : −32,5 dB de résiduel
médian sous `beta = 0,05`, −26,6 dB au-dessus de 0,2 (voir « Le biais propre de
Quegan »). Et un seul indice faux dans les formes fermées laisse la
campagne rendre `-20.10 dB`, identique à la décimale, sans rien signaler.

**Elle ne permet pas non plus de comparer des méthodes entre elles.** Chaque
méthode se jugeant elle-même, une méthode qui ne corrige rien se relit, ne trouve
aucun crosstalk et obtient la note parfaite. Mesuré sur un patch synthétique dont
on connaît le vrai `theta` :

| méthode | résiduel auto-jugé | réciprocité après correction |
|---|---|---|
| aucune correction (`theta = 0`) | **−inf dB**, en apparence parfaite | 0,796 |
| correction parfaite (vrai `theta`) | −23,9 dB, en apparence la pire | 1,000 |
| Quegan | −47,8 dB | 1,000 |

Le classement par résiduel est l'inverse de la réalité ; dans cet exemple, la
réciprocité range correctement les méthodes. Le résiduel ne sert donc qu'à
contrôler la cohérence interne d'une méthode. **Les niveaux 2 et 3 sont
nécessaires pour comparer Quegan et le CVNN, mais pas suffisants** : tous deux
sont aveugles à `alpha` et à une direction de crosstalk, et leur indépendance
dépend de la méthode jugée — voir « Ce que les niveaux 2 et 3 ne voient pas ».

#### Niveau 2 — la réciprocité, sans données extérieures

Un radar monostatique émet et reçoit sur la même antenne : `S_HV = S_VH` est une
propriété du **capteur**, pas du modèle de Quegan. Le crosstalk la viole, et
`metrics.reciprocity` mesure la cohérence `|<HV VH*>| / sqrt(<|HV|²><|VH|²>)`,
valant 1 pour une réciprocité parfaite.

Ce qu'elle voit se lit sur les lignes HV et VH du modèle (« Le biais propre de
Quegan », étape 1), en regroupant les termes en `S_X = S_HV = S_VH` :

```
O_HV = z·S_HH + w·S_VV + ((1 + αzw)/√α)·S_X
O_VH = u·S_HH + v·S_VV + √α·(1 + uv/α)·S_X
```

Si `u = α·z` et `v = α·w`, la seconde ligne devient, pixel par pixel :

```
O_VH = α·z·S_HH + α·w·S_VV + √α·(1 + αzw)·S_X = α · O_HV
```

La cohérence vaut alors exactement 1, quel que soit le crosstalk :
`|<O_HV·O_VH*>| = |α|·<|O_HV|²> = sqrt(<|O_HV|²>·<|O_VH|²>)`. Pour la même
raison, elle ignore `alpha` lui-même : multiplier HV et VH par des constantes ne
change pas une cohérence. La réciprocité ne voit donc que les écarts `u − α·z` et
`v − α·w` (`test_la_reciprocite_est_aveugle_a_u_egal_alpha_z_et_v_egal_alpha_w`).

Elle est indépendante de Quegan : ses formes fermées en `u, v, w, z` ne lisent
que des corrélations co-pol/cross-pol, et si son estimation de `alpha` lit
`C23 = <HV·VH*>`, la cohérence est insensible à `alpha`. Cette indépendance
n'est pas acquise pour toute méthode — voir « Ce que les niveaux 2 et 3 ne voient
pas ». C'est le contrôle du §4.4 de Babu et al. (2022), dont les figures 7a et
8a tracent HV contre VH avant et après calibration.

#### Niveau 3 — les réflecteurs en coin, la seule vérité terrain

Un trièdre répond `S = sqrt(sigma) I` (Fore et al., 2015, équation 5) : ses
canaux croisés sont **nuls par construction**. Tout ce qu'on y mesure est donc du
crosstalk, sans hypothèse de modèle. C'est le §4.6 de Babu et sa figure 9.

Au pic, `S_HH = S_VV` et `S_X = 0` : les lignes du modèle donnent exactement

```
O_HH = (1 + vw)·S_HH     O_HV = (z + w)·S_HH     O_VH = (u + v)·S_HH     O_VV = (1 + uz)·S_HH
```

si bien que `metrics.corner_reflector_crosstalk` mesure `max(|z + w|, |u + v|)`,
au facteur `sqrt(|1 + vw|·|1 + uz|)` près
(`test_la_fuite_mesuree_vaut_max_z_plus_w_et_u_plus_v`). Le trièdre ne voit que
ces deux sommes : il est aveugle à `u = −v, z = −w`, et à `alpha`, qui ne
multiplie que la composante croisée `S_X`, nulle ici.

```bash
python src/measure_reflectors.py        # ~4 min, les 5 acquisitions
```

**Localisation.** Le catalogue JPL `data/corner_reflectors/rosamond_<date>.csv`
donne latitude, longitude, azimut et côté de 27 réflecteurs. Le fichier `.llh`
donne la position géographique de chaque pixel de la grille 2x8 : l'inverser par
plus proche voisin, puis multiplier par le nombre de vues — 8 en azimut, 2 en
distance — place chaque réflecteur à 2-5 pixels près. Un raffinement sur le
maximum de `|HH|` fait le reste.

Le référentiel choisi au téléchargement — `instantaneous-crust`, `mean-crust` ou
`tide-free` — n'est consigné ni dans l'en-tête du `.csv` ni dans l'article de
Babu. Il est sans conséquence : les trois diffèrent d'au plus 30 cm, soit
0,13 pixel en distance. Vérifié en déplaçant chaque réflecteur de 30 cm dans les
quatre directions — le pixel localisé reste identique pour les 15.

**Deux filtres indispensables.**

Le réseau de Rosamond comporte **deux jeux orientés à 180°**, un par direction de
vol. Les réflecteurs à 170-175° renvoient une amplitude de 0,1 à 1,5 ; ceux à
349-355°, alignés sur le `Peg Heading` de -10,03° du `.ann`, renvoient 23 à 83.
Sans ce filtre, douze mesures sur vingt-sept seraient du bruit.

Un contrôle de signature écarte ensuite ce qui n'est pas un trièdre :
`|VV/HH|` dans `[0,7 ; 1,4]` et phase co-polaire sous 30°. Les cibles brillantes
non réflectrices de la zone présentent `|VV/HH| ~ 0,2` et des phases de 150°.

Quinze réflecteurs répondent — dix de 2,44 m et cinq de 4,80 m — aux mêmes
positions au pixel près sur les cinq dates. **La mesure ne retient que les huit
que Babu liste dans sa *Data Availability Statement*** : 13, 14, 15, 17, 18, 20,
21 et 22, tous des trièdres de 2,4384 m (8 pieds), la seule taille qu'il utilise
(§2). Les CR 16 et 19, de même taille et qui répondent aussi, sont exclus comme
chez lui, bien que l'article ne justifie pas ce choix. Son §4.6 parle par
ailleurs de « six » réflecteurs : l'article se contredit, on suit la liste
explicite (`REFLECTEURS_BABU`).

**Agrégation.** Babu rapporte une *average crosstalk* par date (§4.6), sans
préciser si la moyenne porte sur les dB ou sur les puissances. Le script moyenne
les dB, lecture la plus directe d'une moyenne annoncée en dB, et affiche la
médiane en contrôle : elle est robuste au CR 15, systématiquement plus fort.

#### Le piège de circularité, et comment il est évité

Mesurer au réflecteur ne suffit pas : encore faut-il que le `theta` appliqué ne
vienne pas de lui.

Le lac sec rétrodiffuse si peu qu'un réflecteur porte **23 à 31 % de la puissance
d'un patch 256x256, dans un seul pixel**. Estimer `theta` sur ce patch revient
donc à l'ajuster sur le réflecteur, puis à s'en servir pour le corriger. Mesuré :
la fuite chutait de 18 dB, et un `theta` pris 512 pixels à côté ne donnait plus
aucune amélioration.

`theta_de_la_bande` lève la circularité par deux précautions :

- la **bande** de Fore §V-B — `±10` colonnes sur toute la hauteur de la fauchée,
  et non un patch ;
- la **moyenne des `theta`** plutôt que de la covariance. C'est le point décisif :
  moyenner `C` laisse un pixel brillant écraser l'estimation, moyenner les
  `theta` donne le même poids à chacune des 1,04 million de fenêtres 5x5. Le
  réflecteur ne pèse plus que `2,4e-05`.

⚠️ **C'est une justification de fond du protocole de Babu §3.3.1 que ni lui ni
Fore n'énoncent**, et elle n'apparaît que sur une scène portant des cibles
ponctuelles. C'est aussi pourquoi `measure_reflectors.py` force `window: [5, 5]`
en dur plutôt que de lire le YAML : `full` y redonnerait des résultats circulaires.

Contrôle : un `theta` estimé 200 colonnes à côté corrige aussi bien, à 1,0-3,0 dB
près et sans biais systématique.

#### Résultats aux réflecteurs

Les huit réflecteurs de Babu, cinq dates, quarante mesures. Fuite co-pol vers
cross-pol, **moyenne** par date comme chez Babu (§4.6) :

| date | n | avant | après | réduction | Babu, avant | Babu, après\* |
|---|---|---|---|---|---|---|
| 11 février | 8 | −24,27 dB | −36,93 dB | 12,67 dB | −21,62 dB | −32,89 dB |
| 20 mai | 8 | −23,84 dB | **−33,18 dB** | 9,34 dB | **> −20 dB**† | **−27,51 dB** |
| 10 juin | 8 | −23,78 dB | −40,07 dB | 16,28 dB | −20,82 dB | −30,47 dB |
| 16 août | 8 | −23,61 dB | −38,63 dB | 15,01 dB | −21,70 dB | −30,85 dB |
| 20 septembre | 8 | −24,27 dB | −35,30 dB | 11,03 dB | −21,88 dB | −30,29 dB |
| **ensemble** | **40** | **−23,95 dB** | **−36,82 dB** | **12,87 dB** | — | −30,40 dB |

En médiane, contrôle robuste au CR 15 : −24,07 → −36,33 dB, soit 12,26 dB.

\* colonne « Babu, après » = son **improved** Quegan (§4.6, figure 9). Son
Quegan simple, le seul comparable à ce projet, n'est donné que par la table 4 :
−22,49 → −25,33 dB, soit 2,84 dB.

† Le texte ne donne pour le 20 mai que « greater than −20 dB » ; la moyenne
d'ensemble de la colonne « Babu, avant » n'est donc pas calculable.

**Ce qui concorde.** Le 20 mai ressort comme la pire date des deux côtés après
correction : −33,18 dB ici, −27,51 dB chez lui. La corrélation de rang entre les
deux colonnes « après » vaut `rho_s = +0,60` (Pearson `+0,58`), sur cinq points
seulement : ce n'est pas significatif, c'est une tendance.

**Ce qui diffère.** Le Quegan simple de ce projet réduit la fuite de 12,87 dB,
contre 2,84 dB pour celui de Babu, et dépasse même son improved Quegan (−36,82
contre −30,40 dB en moyenne). Trois causes candidates ont été testées et
écartées :

| hypothèse | test | effet sur la réduction |
|---|---|---|
| taille des réflecteurs (Babu n'utilise que les 2,4 m) | retirer les trièdres de 4,8 m | 11,08 → 11,58 dB (médiane) |
| sélection des réflecteurs (ses 8 identifiants) | retirer les CR 16 et 19 | 11,58 → 12,26 dB (médiane) |
| ordre des corrections (son §3.2 avant le §3.3) | `--desequilibre` | 12,87 → 12,79 dB (moyenne) |

Chacune laisse l'écart intact ou l'**élargit**. Restent deux pistes, invérifiables
sans son code, qui n'est pas public :

1. **Son `alpha` est peut-être faux.** Ses équations 16 à 18 sont celles de Fore
   **non réindexées** — sur un cas de référence, elles donnent `|alpha| = 1,0908`
   au lieu de `1,2990`. Un `alpha` erroné dégrade directement la correction.
2. **Sa mesure n'est pas définie.** L'article ne donne aucune formule de la fuite
   au réflecteur ; celle d'ici est `max(|HV|, |VH|) / sqrt(|HH| |VV|)` au pixel de
   pic. La table 4 n'est en outre rattachée à aucune date : son « après »
   amélioré (−30,85 dB) est celui du 16 août au §4.6, mais son « avant »
   (−22,49 dB) ne correspond à aucune des cinq dates.

Le « avant », lui, est quasi constant d'une date à l'autre — 0,66 dB d'étendue,
contre 1,06 dB chez Babu sur les quatre dates chiffrées — ce qui est cohérent avec
un crosstalk instrumental stable.

#### Le déséquilibre de canaux (Babu §3.2)

```bash
python src/measure_reflectors.py --desequilibre     # ~6 min
```

Avant le crosstalk, le script estime alors `f` et `phi_t + phi_r` aux pics des
réflecteurs (Babu, équations 3 et 5), `g` et `phi_t - phi_r` sur l'image entière
(équations 4 et 6), puis applique la correction diagonale de son équation 7 à la
bande comme au réflecteur. La colonne « avant » reste mesurée sur les données
brutes, comme la colonne *Uncalibrated* de sa table 4.

| date | f | g | phi_t + phi_r | phi_t − phi_r | \|alpha\| sans → avec |
|---|---|---|---|---|---|
| 11 février | 1,011 | 0,923 | −1,6° | −3,6° | 1,241 → 0,981 |
| 20 mai | 1,015 | 0,930 | **−16,3°** | −19,6° | 1,256 → 1,011 |
| 10 juin | 1,002 | 0,909 | +0,2° | −1,9° | 1,220 → 0,908 |
| 16 août | 1,010 | 0,920 | +0,8° | −0,1° | 1,172 → 0,905 |
| 20 septembre | 0,999 | 1,030 | −1,6° | −1,3° | 1,218 → **1,359** |

**L'ordre des corrections ne change rien** : 12,87 → 12,79 dB en moyenne, 12,26 dB
en médiane dans les deux cas. Trois raisons :

- `f` vaut 1 à 1,5 % près, ce qu'on attend d'un produit déjà étalonné par JPL ;
  il s'annule de toute façon dans la mesure au réflecteur ;
- `g` et `phi_t - phi_r` sont absorbés par le `alpha` de Quegan (Fore, figure 7) :
  sur les deux premières dates, `|alpha|` retombe de 1,25 à 1,0 environ une fois
  `g` corrigé en amont — le déséquilibre change seulement de place ;
- même le biais de phase de −16,3° du 20 mai laisse sa réduction inchangée.

**Babu n'est pas reproduit sur `f`.** Il annonce un écart de 0,394 au déséquilibre
co-polaire (table 3), contre 0,015 au plus ici, sur la même pile JPL. Les écarts
sur `g` (0,03 à 0,09 contre 0,128) et sur la phase (environ −2° contre −1,14°,
hors 20 mai) sont du même ordre que les siens.

**Un `g` global ne suffit pas.** Le `|alpha|` local dépend de la colonne de
distance — le 20 septembre, il passe de 1,40 à 1,10 entre les colonnes 4369 et
2345, le 11 février de 1,31 à 1,21 —, sans que le sens soit le même à toutes les
dates : le 16 août, il croît légèrement de 1,17 à 1,20. Ce jour-là, le `g` de
l'image entière (1,030) contredit le `alpha` de la bande (1,218) et l'aggrave
(1,359). C'est un argument pour le modèle de Fore, qui estime `alpha` par ligne de
distance, contre une correction globale.

#### Ce que les niveaux 2 et 3 ne voient pas

**Une direction commune et `alpha`.** La réciprocité voit `u − α·z` et `v − α·w`,
le trièdre `u + v` et `z + w`. Annuler les quatre à la fois laisse une direction :

```
u = α·z,  v = α·w,  u = −v,  z = −w     ⟹     (u, v, w, z) = z·(α, −α, −1, 1)
```

Ni l'un ni l'autre ne voit non plus `alpha`. La même structure vaut, au premier
ordre, pour l'erreur `θ̂ − θ` qui reste après correction. Mesuré sur un patch
synthétique sans bruit (512×512, `beta = 0,15`, `alpha = 1,25·e^{−23°i}`,
`|c| = 0,03`) :

| erreur `θ̂ − θ` | `max \|θ̂ − θ\|` | `1 − réciprocité` | fuite au trièdre |
|---|---|---|---|
| `c·(α, −α, −1, 1)` | −28,5 dB | 1,3·10⁻⁷ | −80 dB |
| `c·(1, 1, 1, 1)` | −30,5 dB | 1,8·10⁻³ | −23,4 dB |
| `c·(1, 0, 0, 0)` | −30,5 dB | 2,4·10⁻³ | −31,4 dB |
| `alpha` faux de 20 % | — | 0 | inchangée |

Une méthode dont l'erreur porte sur cette direction ou sur `alpha` obtient donc,
sur données réelles, la note d'une méthode parfaite.

**Une réciprocité de faible résolution.** Même dans une direction visible, une
erreur de −30 dB n'abaisse la cohérence que d'environ 2·10⁻³. Or, sur la pile,
le bruit thermique la tient à 0,821 avant correction et 0,846 après : sur
synthétique, avec le crosstalk de référence, un SNR cross-pol de 10 dB la
plafonne à 0,897, et de 5 dB à 0,750. Départager deux méthodes à quelques dB
l'une de l'autre demande des statistiques appariées, sur les mêmes patchs.

**Une indépendance relative à la méthode.** La réciprocité ne juge pas Quegan
avec ses propres critères, mais elle jugerait Ainsworth avec les siens. Chez
Babu (équations 23 à 28), `A = (C31 + C21)/2` et `B = (C34 + C24)/2`, si bien
que le second membre `X` est fait de `C31 − C21` et `C34 − C24` : le désaccord
entre HV et VH dans leurs corrélations avec HH et VV. Il en irait de même d'un
réseau entraîné avec une perte de réciprocité.

**Ce qui fermerait l'angle mort.** Un dièdre à 0°, `S = diag(1, 0, 0, −1)`,
répondrait `O_HV = (z − w)·S_HH` et `O_VH = (u − v)·S_HH`. Avec le trièdre, les
quatre termes seraient séparés. Le réseau de Rosamond ne compte que des trièdres,
et `alpha` ne s'y mesure que sur cibles distribuées, comme le font Quegan et
l'équation 4 de Babu. La direction commune et `alpha` ne peuvent donc être jugés
que sur données synthétiques, où `theta` est connu (`metrics.parameter_errors`).

### Tests

```bash
python -m pytest tests/ -q        # 239 tests
```

Trois niveaux, du plus faible au plus fort :

1. **cohérence interne** — `Sigma @ D = I`, aller-retour `s -> D s -> theta` ;
2. **ancrage sur la source** — `D` et `Sigma` comparées aux équations 13 et 15
   de Fore écrites à la lettre. Une même erreur commise dans `D` et dans `Sigma`
   échapperait au niveau 1 ;
3. **invariants de convention** — `POL_ORDER == sorted(POL_ORDER)`, cohérence
   des indices `IDX_*` de `synthetic.py` avec `POL_ORDER`.

Plusieurs conventions sont en outre verrouillées par **sabotage** : on introduit
volontairement l'erreur, on vérifie qu'un test échoue, puis on restaure. Par
exemple, intervertir `phi_t` et `phi_r` dans `ChannelImbalance.correction_matrix`
fait échouer 3 des 19 tests de `test_imbalance.py`.

Si des échecs ne correspondent pas au code que vous lisez, purgez le bytecode :

```bash
find src tests -name __pycache__ -type d -exec rm -rf {} +
```

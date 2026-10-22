# Crosstalk Estimation

Estimation du crosstalk polarimétrique sur la pile UAVSAR Rosamd_35012_04, en
suivant le protocole de Babu et al. (2022) sous une forme simplifiée : **une
seule covariance par patch**, au lieu de ses fenêtres glissantes 5×5. Le projet
fournit les **baselines statistiques** (méthode de Quegan) et leur **validation
aux réflecteurs en coin** ; un encodeur à valeurs complexes (CVNN) leur sera
ensuite comparé.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Carte du projet : des `.slc` aux résultats

Le projet a **deux points d'entrée**, qui partagent le paquet `src/crosstalk/` :

- [src/run_baselines.py](src/run_baselines.py) applique Quegan patch par patch,
  sur la partition de test. C'est la **baseline** que le CVNN devra battre. Les
  patchs lui viennent de [src/patchs.py](src/patchs.py), seul module du projet
  qui dépende de `cvnn`.
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
                     └──► covariance → quegan → matrix  ◄────┘
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
étalonnés en gain et en phase par JPL (`Polarimetrically Calibrated = Yes`),
mais **pas corrigés du crosstalk** (`Cross Talk Removed = No`). Le modèle s'écrit `s' = D s + n`, où `s` est le signal vrai : la
chaîne estime `D` puis corrige, `s_chapeau = Sigma s'`. La distorsion n'est
jamais appliquée aux données réelles.

### Chaîne A — `python src/run_baselines.py`

Le guide d'audit [audit/audit_chaine_A.md](audit/audit_chaine_A.md) détaille
chaque étape sur les plans informatique et mathématique, avec les tests qui la
verrouillent.

| # | étape | code | entrée → sortie |
|---|---|---|---|
| A1 | lire la configuration | `run_baselines.main` | YAML → `dict` |
| A2 | instancier les méthodes | `run_baselines.build_estimators` → `estimators/registry.create_from_config` | `{name: quegan}` → `QueganEstimator()`, que le décorateur `@register_estimator` a inscrit sous le nom `quegan` |
| A3 | fixer le hasard, compléter la configuration | `patchs.build_dataloader` → `cvnn.utils.set_seed(38)`, puis `cvnn/data.validate_and_correct_config` | ajoute `type: polsar`, `has_labels: False`, `num_channels: 4`, lus dans `cvnn/dataset_registry.py` |
| A4 | découvrir les fichiers | `cvnn/data._create_dataset` → `torchcvnn.SLCDataset` | groupe les `.slc` par date, lit les `.ann` ; grille de 958 × 154 = **147 532 patchs de 64 px par date**, 737 660 au total. Aucun pixel n'est encore lu |
| A5 | **découper train / valid / test** | `cvnn/data._groupes_par_emplacement`, appelée par `get_dataloaders` | mélange les 147 532 emplacements au sol, les 5 dates suivent leur emplacement → 590 130 / 73 765 / 73 765 patchs, sans fuite |
| A6 | construire les transformations | `cvnn/transform_registry.build_transform_pipeline` | empilement du `dict` en ordre alphabétique (HH, HV, VH, VV) = `POL_ORDER`, tenseur, puis `GlobalScalarNormalize` qui divise par 0,16444 = √(2 × 0,013521) |
| A7 | construire les chargeurs | `cvnn/data.get_dataloaders` | seul `test_loader` sert : `shuffle=False`, lots de 16, soit 4 611 lots |
| A8 | lire les patchs | `patchs.iter_patches` | lot `(16, 4, 64, 64)` `complex64` → un patch numpy `(4, 64, 64)` à la fois |
| A9 | boucle d'évaluation | `run_baselines.evaluate_patches` | une ligne par couple patch/méthode ; une `CrosstalkError` donne le statut `ill_conditioned` sans arrêter la boucle |
| A10 | estimer, corriger, ré-estimer | `metrics.residual_crosstalk` | détaillé en A10a-d ci-dessous |
| A10a | covariance | `covariance.polarimetric_covariance` | `C = (1/N) Σ k·kᴴ` sur les **4 096 pixels** du patch → une seule `C` de forme `(4, 4)` `complex128` |
| A10b | formes fermées | `QueganEstimator.solve`, `QueganEstimator.estimate` | `Δ`, `u, v, w, z` (Babu, équation 13), `X`, `alpha` → un `CrosstalkParameters` par patch ; aucune validation : un patch dégénéré (constant, nul, canal éteint, HH et VV parfaitement corrélés) rend des `nan` ou des valeurs finies sans signification, cas absent des 73 765 patchs de test |
| A10c | correction | `CrosstalkEstimator.estimate_and_correct` → `correct(patch, theta)` → `matrix.apply_correction` → `sigma_calibration_matrix`, `apply_matrix` | `Sigma` `(4, 4)`, protégée contre `(uw−1)(vz−1) ≈ 0`, puis `patch_corrige = Sigma · s'` pour chaque pixel ; rend `(patch_corrige, theta_initial)` |
| A10d | ré-estimation et réciprocité | `estimate(patch_corrige)`, `metrics.reciprocity` | `ResidualReport` : `theta_initial`, `theta` résiduel, réciprocité avant et après ; `evaluate_patches` le reçoit sous le nom `residual_report_patch` |
| A11 | synthèse | `run_baselines.summarize` | la médiane des 73 765 lignes → tableau affiché ; rien n'est écrit sur disque |

### Chaîne B — `python src/measure_reflectors.py`

Ni cvnn ni normalisation : les `.slc` sont ouverts en `numpy.memmap` et lus aux
seuls endroits utiles.

| # | étape | code | ce qui se passe |
|---|---|---|---|
| B1 | pour chaque date | `main` → `mesure_une_acquisition` | ouvre 4 memmaps `(61 349, 9 874)` et le `.llh` |
| B2 | **localiser les réflecteurs** | `localise_les_reflecteurs` | lit le catalogue (`lire_catalogue`) ; garde les réflecteurs orientés vers le radar (`oriente_vers_le_radar`, ±45° du cap) et les 8 de Babu (`est_retenu_par_babu`) ; les place par plus proche voisin dans le `.llh`, ×8 en azimut et ×2 en distance (`position_radar`) ; cherche le pic de `\|HH\|` à ±60 px (`affine_sur_le_pic`) ; contrôle la signature de trièdre (`est_un_triedre`) |
| B3 | **`theta` de la bande** | `theta_de_la_bande` | colonnes `c ± 10` sur les 61 349 lignes, environ 40 Mo, en **une seule covariance**. La bande dilue le réflecteur sans l'exclure : la mesure « après » en est optimiste d'environ 1,5 dB |
| B4 | mesurer | `mesure_un_reflecteur` → `metrics.corner_reflector_crosstalk` | « avant » = `20 log10(max(\|HV\|, \|VH\|) / sqrt(\|HH\| \|VV\|))` au pic brut ; « après » = la même mesure sur `Sigma · s'`, où `s'` est le pic brut |
| B5 | agréger | `agrege`, `resume` | tableau par date, moyenne (Babu) et médiane, puis l'ensemble des 40 mesures |

### Ordre de lecture recommandé

Des fondations vers les points d'entrée : chaque fichier ne dépend que de ceux
qui le précèdent.

| # | fichier | à vérifier en le lisant |
|---|---|---|
| 1 | [configs/data_uavsar.yaml](configs/data_uavsar.yaml) | `seed`, `patch_size`, ratios, bloc `baselines:` |
| 2 | [cvnn/src/cvnn/data.py](cvnn/src/cvnn/data.py) — `_create_dataset`, `_groupes_par_emplacement`, `get_dataloaders` | le découpage groupé par emplacement ; le test chargé sans mélange |
| 3 | [cvnn/src/cvnn/transform_registry.py](cvnn/src/cvnn/transform_registry.py) — `GlobalScalarNormalize` | la normalisation est **un seul scalaire réel**, donc `theta` invariant |
| 4 | [src/crosstalk/exceptions.py](src/crosstalk/exceptions.py) | le seul cas d'échec : D singulière |
| 5 | [src/crosstalk/parameters.py](src/crosstalk/parameters.py) | ordre `(u, v, w, z, alpha)`, conversions en dB |
| 6 | [src/crosstalk/matrix.py](src/crosstalk/matrix.py) | `POL_ORDER`, D réindexée de Fore, `Sigma · D = I`, `apply_matrix` |
| 7 | [src/crosstalk/covariance.py](src/crosstalk/covariance.py) | une seule `C` par patch, forme `(4, 4)` ; convention `C_ij = <S'_i S'_j*>` |
| 8 | [src/crosstalk/estimators/estimator.py](src/crosstalk/estimators/estimator.py), puis [registry.py](src/crosstalk/estimators/registry.py) | interface abstraite, enregistrement par décorateur |
| 9 | [src/crosstalk/estimators/quegan.py](src/crosstalk/estimators/quegan.py) | formes fermées, sans validation de la covariance |
| 10 | [src/crosstalk/metrics.py](src/crosstalk/metrics.py) | le résiduel est **circulaire**, la réciprocité ne l'est pas ; formule au réflecteur |
| 11 | [src/patchs.py](src/patchs.py) | patchs de test servis par `cvnn`, lots dépliés en patchs |
| 12 | [src/run_baselines.py](src/run_baselines.py) | assemblage de la chaîne A |
| 13 | [src/measure_reflectors.py](src/measure_reflectors.py) | assemblage de la chaîne B ; la bande comme rempart, partiel, contre la circularité |
| 14 | [src/crosstalk/synthetic.py](src/crosstalk/synthetic.py) et [tests/](tests/) | hors chaîne : ne servent qu'aux tests, à `theta` connu |

**Trois points à garder en tête en relisant.**

- **L'ordre des canaux est implicite** : il naît en A6, de l'ordre alphabétique
  du `dict` rendu par torchcvnn, et tout le reste en dépend. Un test verrouille
  `POL_ORDER == sorted(POL_ORDER)`.
- **Les deux chaînes n'ont pas la même normalisation**, sans effet : `theta` et la
  fuite au réflecteur sont des rapports.
- **La bande de la chaîne B contient le réflecteur qu'elle sert à corriger** :
  la mesure « après » est optimiste d'environ 1,5 dB.

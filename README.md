# Crosstalk Estimation

Estimation du crosstalk sur images SAR par apprentissage profond.

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
| A8 | lire les patchs | `run_baselines.iter_patches` | lot `(16, 4, 256, 256)` `complex64` → un patch numpy `(4, 256, 256)` à la fois |
| A9 | boucle d'évaluation | `run_baselines.evaluate_patches` | une ligne par couple patch/méthode ; une `CrosstalkError` donne le statut `ill_conditioned` sans arrêter la boucle |
| A10 | estimer, corriger, ré-estimer | `metrics.residual_crosstalk` | détaillé en A10a-e ci-dessous, et tracé sur un patch réel dans « Chaîne de traitement » plus bas |
| A10a | covariance | `covariance.polarimetric_covariance` | 16 produits `O_i O_j*`, table de sommes cumulées → `C` de forme `(252, 252, 4, 4)` `complex128`, soit **63 504 fenêtres 5×5** |
| A10b | formes fermées | `QueganEstimator._solve` | par fenêtre : `Δ`, `u, v, w, z` (Babu, équation 13), `X`, `alpha` ; masque `valides` : rapports finis et au-dessus des seuils |
| A10c | moyenne | `QueganEstimator.estimate` | moyenne des `theta` sur les fenêtres valides → un `CrosstalkParameters` par patch |
| A10d | correction | `CrosstalkEstimator.correct` → `physics.inverse_distortion_matrix` → `physics.apply_matrix` | `Sigma` `(4, 4)`, protégée contre `(uw−1)(vz−1) ≈ 0`, puis `s_chapeau = Sigma · o` pour chaque pixel |
| A10e | ré-estimation et réciprocité | `estimate(s_chapeau)`, `metrics.reciprocity` | `ResidualReport` : initial, résiduel, réciprocité avant et après |
| A11 | synthèse | `run_baselines.summarize` | médiane de chaque colonne sur les 4 540 lignes → tableau affiché. **Rien n'est écrit sur disque** |

### Chaîne B — `python src/measure_reflectors.py [--desequilibre]`

Ni cvnn ni normalisation : les `.slc` sont ouverts en `numpy.memmap` et lus aux
seuls endroits utiles.

| # | étape | code | ce qui se passe |
|---|---|---|---|
| B1 | pour chaque date | `main` → `mesure_une_acquisition` | ouvre 4 memmaps `(61 349, 9 874)` et le `.llh` |
| B2 | **localiser les réflecteurs** | `localise_les_reflecteurs` | lit le catalogue (`lire_catalogue`) ; garde les réflecteurs orientés vers le radar (`oriente_vers_le_radar`, ±45° du cap) et les 8 de Babu (`est_retenu_par_babu`) ; les place par plus proche voisin dans le `.llh`, ×8 en azimut et ×2 en distance (`position_radar`) ; cherche le pic de `\|HH\|` à ±60 px (`affine_sur_le_pic`) ; contrôle la signature de trièdre (`est_un_triedre`) |
| B3 | déséquilibre de canaux, avec `--desequilibre` seulement | `desequilibre_de_l_acquisition` → `crosstalk/imbalance.py` | `f` et `phi_t + phi_r` aux 8 pics ; `g` et `phi_t - phi_r` sur toute l'image, par tranches de 2 048 lignes → matrice diagonale `K` |
| B4 | **`theta` de la bande** | `theta_de_la_bande` | colonnes `c ± 10` sur les 61 349 lignes, en tranches de 8 192 (`K` appliquée si présente) ; Quegan 5×5 par tranche, puis moyenne des `theta`. C'est ce qui rend la mesure non circulaire |
| B5 | mesurer | `mesure_un_reflecteur` → `metrics.corner_reflector_crosstalk` | « avant » = `20 log10(max(\|HV\|, \|VH\|) / sqrt(\|HH\| \|VV\|))` au pic brut ; « après » = la même mesure sur `Sigma · (K ·) o` |
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
- **La chaîne A ne sauvegarde rien** : [experiments/baselines.csv](experiments/baselines.csv)
  date d'une version antérieure du code.
- **Les deux chaînes n'ont pas la même normalisation**, sans effet : `theta` et la
  fuite au réflecteur sont des rapports.
- **La fenêtre 5×5 est fixée en dur dans la chaîne B** (`main` de
  `measure_reflectors.py`), et non lue dans le YAML : `full` y rendrait la mesure
  circulaire — voir « Le piège de circularité » plus bas.

## Pipeline de données

Entièrement assuré par les deux frameworks existants, sans code spécifique au
projet :

- [`torchcvnn`](https://github.com/torchcvnn/torchcvnn) — `SLCDataset` découvre
  les `.slc`, lit les `.ann`, découpe la grille de patchs et rend un dictionnaire
  `{HH, HV, VH, VV}` ; la transform `PolSAR` l'empile en `(4, H, W)` complexe.
- [`cvnn`](https://github.com/QuentinGABOT/cvnn) — `get_dataloaders`
  découpe train / validation / test et construit les `DataLoader`.

Données attendues : `data/Rosamd_35012_04/`, pile UAVSAR *Stack SLC* de
5 acquisitions × 4 polarisations, 61 349 × 9 874 pixels `complex64` par fichier
(4,85 Go chacun, ~92 Go au total).

```python
import sys, yaml
sys.path.insert(0, "cvnn/src")
from cvnn.data import get_dataloaders, validate_and_correct_config

cfg = validate_and_correct_config(yaml.safe_load(open("configs/data_uavsar.yaml")))
train_loader, valid_loader, test_loader = get_dataloaders(cfg, use_cuda=True)

batch = next(iter(train_loader))    # (B, 4, H, W) complex64, canaux [HH, HV, VH, VV]
```

Tout se règle dans [configs/data_uavsar.yaml](configs/data_uavsar.yaml) :
taille et pas des patchs, ratios de découpe, `batch_size`, `num_workers`,
transforms.

### Branchement dans cvnn

`UavsarSLC` a été ajouté au framework comme n'importe quel autre dataset, en
deux endroits :

- `cvnn/src/cvnn/dataset_registry.py` — entrée `UavsarSLC`
  (`type: polsar`, 4 canaux, sans étiquettes) ;
- `cvnn/src/cvnn/data.py` — import de `SLCDataset`, branche
  `elif dataset_name == "UavsarSLC"` dans `_create_dataset`, et `UavsarSLC`
  ajouté à la liste des datasets à `patch_size`.

ℹ️ Contrairement à ce qu'indiquait une version antérieure de ce fichier, `cvnn/`
**est suivi par Git dans ce dépôt** (136 fichiers) : le `.gitignore` ne contient
aucune règle le concernant. Ces deux modifications sont donc versionnées ici, et
il reste souhaitable de les reporter en amont dans le dépôt `cvnn`.

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
`s = D·s_vrai` devient `Λ·s = (Λ·D)·s_vrai`, et les paramètres de crosstalk,
qui sont des **rapports** de coefficients de `D`, sont multipliés par σⱼ/σᵢ. Il
faut alors appliquer Λ⁻¹ pour revenir à la grandeur physique.

Sous une homothétie **globale** `α`, au contraire, `D` devient `α·D` et ces
rapports sont invariants : la sortie du réseau est directement la grandeur
physique. La configuration livrée obtient cela **sans code**, en donnant la
*même* covariance aux 4 canaux (`cov_complex_value`). L'échelle est standardisée
(module moyen ~0,38) et les rapports entre polarisations restent exacts.

Si vous remettez `recompute_statistics: True`, `cvnn` réécrit ces valeurs avec
une covariance par canal et le comportement du tableau ci-dessus revient.

## Estimation du crosstalk

Paquet `src/crosstalk/`, indépendant du réseau : il fournit les **baselines
statistiques** contre lesquelles le CVNN sera comparé. Seule la méthode de
Quegan est implémentée à ce jour.

```bash
python src/run_baselines.py        # depuis la racine du dépôt
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

À titre de repère, Babu et al. (2022, table 4) annoncent **-22,49 dB** non
calibré. Son résiduel après Quegan, **-25,33 dB**, n'est pas comparable au nôtre :
il le mesure aux réflecteurs en coin (§4.6), là où `residual_crosstalk` ré-estime
sur les données corrigées — voir l'avertissement en tête de `metrics.py`.

La fenêtre glissante coûte environ 12× le mode `full` en temps de calcul. Sur
cette pile, elle déplace le crosstalk de -20,1 à -21,2 dB et `|alpha|` de 1,187 à
1,226 ; cette hausse est un effet de Jensen — moyenner 63 504 estimations
bruitées d'une quantité positive la surestime.

Tout se règle dans le bloc `baselines:` de
[configs/data_uavsar.yaml](configs/data_uavsar.yaml).

### Architecture

| module | rôle |
|---|---|
| `parameters.py` | le jeu immuable `theta = (u, v, w, z, alpha)` |
| `exceptions.py` | `CrosstalkError` et ses deux sous-classes |
| `physics.py` | `POL_ORDER`, la matrice de distorsion `D`, son inverse `Sigma`, leur application |
| `covariance.py` | `C = <o o^H>`, plein patch ou fenêtre glissante |
| `metrics.py` | erreurs sur `theta` (synthétique) et crosstalk résiduel (réel) |
| `imbalance.py` | déséquilibre de canaux `(f, g, phi_t, phi_r)` de Babu §3.2, et sa correction diagonale |
| `synthetic.py` | signal vrai conforme aux hypothèses de Quegan, pour les tests |
| `estimators/` | ce qui distingue les méthodes : l'estimation de `theta`, et le registre |

La frontière est stricte : le modèle physique et la covariance sont **communs**
à toutes les méthodes ; une méthode nouvelle n'écrit que `estimate`.

### Chaîne de traitement

Les `.slc` sortent de l'instrument **déjà distordus** — `Cross Talk Removed = No`
dans les `.ann`. Ils sont en revanche déjà étalonnés en gain et en phase par le
processeur JPL — `Polarimetrically Calibrated = Yes`, sur les 20 `.ann` de la
pile — ce que confirme la mesure du déséquilibre de canaux (voir « Résultats aux
réflecteurs »). La chaîne ne fait donc qu'estimer puis corriger le crosstalk ; la
distorsion n'est jamais appliquée aux données réelles.

| # | opération | code | entrée → sortie |
|---|---|---|---|
| 1 | lire les 4 `.slc` | `torchcvnn.SLCDataset` | 4 fichiers 61 349 × 9 874 `complex64` |
| 2 | empiler en ordre alphabétique | transform `PolSAR` | `o = (4, 256, 256)` |
| 3 | normaliser | `GlobalScalarNormalize` | `o / 0.16444` |
| 4 | **covariance**, fenêtre glissante 5×5 | `covariance.polarimetric_covariance` | `C = (252, 252, 4, 4)` `complex128` |
| 5 | **formes fermées** par fenêtre, puis moyenne | `estimators.quegan.QueganEstimator._solve` | `theta = (u, v, w, z, alpha)` |
| 6 | **correction** | `physics.apply_matrix(Sigma, o)` | `s_chapeau = (4, 256, 256)` |
| 7 | ré-estimer sur `s_chapeau` | `estimate` | crosstalk résiduel |

Tracé sur le patch de contrôle `(30000, 4500)` de l'acquisition du 11 février :

```
ÉTAPE 1   HH 0.007157   HV 0.000731   VH 0.001004   VV 0.007269   (puissances moyennes)
ÉTAPE 4   C de forme (252, 252, 4, 4) : 63 504 fenêtres 5×5, une matrice chacune
          fenêtres retenues : 63 504 / 63 504
ÉTAPE 5   moyenne des 63 504 theta
          u =  0.024319 + 0.005222j    fuite HH -> VH
          z =  0.046798 - 0.025522j    fuite HH -> HV
          v = -0.072366 - 0.033255j    fuite VV -> VH
          w = -0.013079 - 0.015735j    fuite VV -> HV
          |alpha| = 1.290804           déséquilibre HV / VH
          crosstalk = -21.98 dB
ÉTAPE 7   résiduel = -32.59 dB         réduction 10.62 dB
```

Les étapes 4 à 7 tournent sur chacun des 4 540 patchs de la partition de test
(908 emplacements au sol × 5 dates, le découpage groupant les dates d'un même
emplacement), et `summarize` prend la médiane de chaque colonne.

**Le crosstalk se lit dans la covariance, pas dans les pixels.** L'étape 4 réduit
les 25 pixels × 4 canaux de chaque fenêtre à une matrice 4×4. Toute
l'information y est : Quegan suppose que sur le signal **vrai**, les termes co-pol/cross-pol
`<HH·VH*>` et `<VV·HV*>` sont **nuls** par symétrie de réflexion (équation 17 de
Fore). Tout ce qu'on mesure à ces positions sur le signal **observé** est donc du
crosstalk — c'est exactement ce que lisent les formes fermées.

**La normalisation ne change pas `theta`.** `GlobalScalarNormalize` divise par un
scalaire réel unique. Sous une homothétie globale `g`, `D` devient `g·D`, et
`(u, v, w, z)` sont des **rapports** de coefficients de `D`, donc invariants. Une
normalisation *par canal* les changerait ; celle qui est configurée ne le fait
pas — voir « Point d'attention : la normalisation » plus haut.

**`apply_distortion` ne tourne jamais sur les données réelles.** Elle sert
uniquement aux tests, pour fabriquer une observation à `theta` connu — le seul
moyen de disposer d'une vérité terrain, puisque les `.slc` n'en ont pas. C'est
ainsi qu'est mesuré le biais résiduel de 1 % du modèle, sans bruit et à un
million d'échantillons : le terme `O(|theta|²)` que Quegan néglige.

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

**Fore est la référence du code**, parce que c'est le seul des deux qui forme un
système fermé : `Sigma_eq15 @ D_eq13 = I` à `2.2e-16`, et ses équations 24-26
retrouvent exactement le `theta` injecté par son équation 13.

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
| éq. 16 `C_ij = <O_i O_j*>` | identique |
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

**Règle suivie** : Babu, sauf là où ses équations sont incohérentes ou
incomplètes, auquel cas Fore.

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
sur le même plancher. Et un seul indice faux dans les formes fermées laisse la
campagne rendre `-20.10 dB`, identique à la décimale, sans rien signaler.

Cette métrique sert à **comparer des méthodes entre elles**, jamais à annoncer
une performance absolue.

#### Niveau 2 — la réciprocité, indépendante et sans données extérieures

Un radar monostatique émet et reçoit sur la même antenne : `S_HV = S_VH` est une
propriété du **capteur**, pas du modèle de Quegan. Le crosstalk la viole, et
`metrics.reciprocity` mesure la cohérence `|<HV VH*>| / sqrt(<|HV|²><|VH|²>)`,
valant 1 pour une réciprocité parfaite.

Elle est indépendante parce que `estimate` ne lit que des corrélations
co-pol/cross-pol, jamais l'accord entre HV et VH. C'est le contrôle du §4.4 de
Babu et al. (2022), dont les figures 7a et 8a tracent HV contre VH avant et
après calibration.

#### Niveau 3 — les réflecteurs en coin, la seule vérité terrain

Un trièdre répond `S = sqrt(sigma) I` (Fore et al., 2015, équation 5) : ses
canaux croisés sont **nuls par construction**. Tout ce qu'on y mesure est donc du
crosstalk, sans hypothèse de modèle. C'est le §4.6 de Babu et sa figure 9.

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

### Tests

```bash
python -m pytest tests/ -q        # 206 tests
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

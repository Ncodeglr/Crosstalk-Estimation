# Crosstalk Estimation

Estimation du crosstalk sur images SAR par apprentissage profond.
Projet de recherche — SONDRA Lab, CentraleSupélec.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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
         méthode  patchs  échecs  initial dB  résiduel dB  réduction dB  |alpha|
--------------------------------------------------------------------------------
          quegan    4541       0       -21.2        -30.7           9.5    1.226
```

Chaque colonne est la **médiane** sur les patchs, un `theta` étant estimé par
patch. Ces valeurs dépendent de `seed:`, qui fixe le découpage train / valid /
test : changer la graine change les patchs évalués, pas la méthode. Durée :
6 min 33 s.

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
| `synthetic.py` | signal vrai conforme aux hypothèses de Quegan, pour les tests |
| `estimators/` | ce qui distingue les méthodes : l'estimation de `theta`, et le registre |

La frontière est stricte : le modèle physique et la covariance sont **communs**
à toutes les méthodes ; une méthode nouvelle n'écrit que `estimate`.

### Chaîne de traitement

Les `.slc` sortent de l'instrument **déjà distordus** — `Cross Talk Removed = No`
dans les `.ann`. La chaîne ne fait donc qu'estimer puis corriger ; la distorsion
n'est jamais appliquée aux données réelles.

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

Les étapes 4 à 7 tournent sur chacun des 4541 patchs de la partition de test, et
`summarize` prend la médiane de chaque colonne.

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

Deux écarts délibérés à son équation 25 : le `4|alpha_2|²` ci-dessus, et
l'absence du facteur de phase `alpha_1/|alpha_1|` — `alpha` est rendu réel
positif, comme à l'équation 15 de Babu.

⚠️ En comparant le code au papier, ne « corrigez » pas `C31` en `C21` : c'est la
réindexation, et `tests/test_physics.py::test_d_est_la_matrice_de_fore_reindexee`
la verrouille.

### Ce qui n'est pas implémenté

Le paquet suit **Babu et al. (2022), §3.3.1** — le Quegan de base. Trois écarts
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

### Tests

```bash
python -m pytest tests/ -q        # 175 tests
```

Trois niveaux, du plus faible au plus fort :

1. **cohérence interne** — `Sigma @ D = I`, aller-retour `s -> D s -> theta` ;
2. **ancrage sur la source** — `D` et `Sigma` comparées aux équations 13 et 15
   de Fore écrites à la lettre. Une même erreur commise dans `D` et dans `Sigma`
   échapperait au niveau 1 ;
3. **invariants de convention** — `POL_ORDER == sorted(POL_ORDER)`, cohérence
   des indices `IDX_*` de `synthetic.py` avec `POL_ORDER`.

Si des échecs ne correspondent pas au code que vous lisez, purgez le bytecode :

```bash
find src tests -name __pycache__ -type d -exec rm -rf {} +
```

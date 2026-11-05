# Crosstalk Estimation

Estimation du crosstalk polarimétrique sur la pile UAVSAR Rosamd_35012_04.
L'objectif est de **comparer la méthode de Quegan à un réseau à valeurs complexes
(CVNN)**, avec **un jeu `(u, v, w, z, alpha)` par patch**, là où Babu et al. (2022)
utilisent des fenêtres glissantes 5×5.

Le dépôt contient :

- les **baselines statistiques** : Quegan (formes fermées) et un solveur exact, qui
  garde les hypothèses de Quegan sans son approximation au premier ordre ;
- le **banc d'essai par réinjection** : un crosstalk connu `theta_k` est ajouté à
  des patchs réels, et deux CVNN apprennent à le retrouver, un MLP sur la covariance
  du patch et un ResNet sur ses pixels.

La comparaison directe n'a pas encore de script : voir
[État du dépôt](#état-du-dépôt-au-2026-11-05).

## Installation

```bash
python3 -m venv .venv       # Python 3.12
source .venv/bin/activate   # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **PyTorch** : `requirements.txt` installe les versions compilées pour CUDA 13.0.
- **cvnn** : versionné dans `cvnn/` et non installé comme paquet ; les scripts
  ajoutent `cvnn/src` au chemin d'import.
- **Données** : la pile n'est pas versionnée et se place dans
  `data/Rosamd_35012_04/` (environ 92 Go).
- **wandb** : les entraînements y envoient leurs courbes. Il faut au choix lancer
  `wandb login`, passer `logging.wandb.mode` à `offline`, ou supprimer la section
  `logging.wandb` du YAML.

## Lancer

```bash
python src/run_baselines.py   # chaîne A : baselines sur les patchs de test bruts
python src/train_mlp.py       # banc d'essai : MLP (accepte un autre YAML en argument)
python src/train_resnet.py    # banc d'essai : ResNet (idem)
pytest tests
```

## Carte du projet : des `.slc` aux résultats

Deux chaînes partagent le paquet `src/crosstalk/`.

```
                    data/Rosamd_35012_04/  (20 .slc + 20 .ann)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
 chaîne A : run_baselines.py              banc d'essai : train_mlp.py, train_resnet.py
 torchcvnn → cvnn → patchs                torchcvnn → cvnn → injection → cache
            │                                                   │
            ▼                                                   ▼
 covariance par patch                     MLP (covariance) ou ResNet (pixels)
 → estimateur → correction                → theta_k estimé
            │                                                   │
            ▼                                                   ▼
 résiduel, réciprocité                    erreur à theta_k connu
 tableau affiché                          logs/<task>_<run>/evaluation.yaml
```

| chaîne | données | ce qui juge | limite |
|---|---|---|---|
| chaîne A | patchs de test bruts | l'estimateur lui-même pour le résiduel ; rien pour la réciprocité | le résiduel est circulaire |
| banc d'essai | patchs réels avec `theta_k` injecté | `theta_k`, connu exactement | mesure le crosstalk **ajouté**, pas le crosstalk total |

**Les chiffres de ces deux chaînes ne se comparent pas entre eux** : le résiduel de
la chaîne A et l'erreur du banc d'essai ne mesurent pas la même grandeur.

Les fonctions sont citées par leur nom plutôt que par leur numéro de ligne, qui
change à chaque modification du code. Les valeurs chiffrées sont celles des YAML au
2026-11-05 : les réglages sont encore en cours.

### Les entrées

| fichier | contenu |
|---|---|
| `*_L090{HH,HV,VH,VV}_04_BC_s1_1x1.slc` | 5 dates × 4 polarisations = 20 fichiers de 61 349 lignes (azimut) × 9 874 colonnes (distance), en `complex64`, soit environ 4,8 Go chacun |
| `*.ann` | dimensions, cap de l'avion (−10,03°), `Polarimetrically Calibrated = Yes`, `Cross Talk Removed = No` |
| [configs/data_uavsar.yaml](configs/data_uavsar.yaml) | chaîne A : taille de patch, graine, ratios du découpage, normalisation, bloc `baselines:` |
| [configs/cvnn_mlp.yaml](configs/cvnn_mlp.yaml), [configs/cvnn_resnet.yaml](configs/cvnn_resnet.yaml) | banc d'essai : réseau, découpage, loi d'injection, optimiseur, journalisation |

**Notation.** Les `.slc` sont le **`s'` de Babu** (éq. 7 et 8), le `O` de Fore :
étalonnés en gain et en phase par JPL (`Polarimetrically Calibrated = Yes`), mais
**pas corrigés du crosstalk** (`Cross Talk Removed = No`). Le modèle s'écrit
`s' = D s + n`, où `s` est le signal vrai : les estimateurs estiment `D` puis
corrigent, `s_chapeau = Sigma s'`. La chaîne A n'applique jamais de distorsion aux
données réelles. Le banc d'essai le fait volontairement, pour créer une cible connue.

### Chaîne A — `python src/run_baselines.py`

| # | étape | code | entrée → sortie |
|---|---|---|---|
| A1 | lire la configuration | `run_baselines.main`, `lire_YAML` | YAML → `dict`, avec `trainpath` ancré à la racine du dépôt |
| A2 | instancier les méthodes | `run_baselines.build_estimators` → `estimators/registry.create_from_config` | `{name: quegan}` → `QueganEstimator()`, que le décorateur `@register_estimator` a inscrit sous le nom `quegan` |
| A3 | fixer le hasard, compléter la configuration | `patchs.build_dataloader` → `cvnn.utils.set_seed(16)`, puis `cvnn/data.validate_and_correct_config` | ajoute `type: polsar`, `has_labels: False`, `num_channels: 4`, lus dans `cvnn/dataset_registry.py` |
| A4 | découvrir les fichiers | `cvnn/data._create_dataset` → `torchcvnn.SLCDataset` | groupe les `.slc` par date, lit les `.ann` ; grille de 479 × 77 = **36 883 patchs de 128 px par date**, 184 415 au total. Aucun pixel n'est encore lu |
| A5 | **découper train / valid / test** | `cvnn/data._groupes_par_emplacement`, appelée par `get_dataloaders` | mélange les 36 883 emplacements au sol, les 5 dates suivent leur emplacement → 147 535 / 18 440 / 18 440 patchs, sans fuite |
| A6 | construire les transformations | `cvnn/transform_registry.build_transform_pipeline` | empilement du `dict` en ordre alphabétique (HH, HV, VH, VV) = `POL_ORDER`, tenseur, puis `GlobalScalarNormalize` qui divise par 0,16444 = √(2 × 0,013521) |
| A7 | construire les chargeurs | `cvnn/data.get_dataloaders` | seul `test_loader` sert : sans mélange, lots de 256, soit 73 lots |
| A8 | lire les patchs | `patchs.iter_patches` | lot `(256, 4, 128, 128)` `complex64` → un patch numpy `(4, 128, 128)` à la fois |
| A9 | boucle d'évaluation | `run_baselines.evaluate_patches` | une ligne par couple patch/méthode ; une `CrosstalkError` donne le statut `ill_conditioned` sans arrêter la boucle |
| A10 | estimer, corriger, ré-estimer | `metrics.residual_crosstalk` | détaillé en A10a-d ci-dessous |
| A10a | covariance | `covariance.polarimetric_covariance` | `C = (1/N) Σ k·kᴴ` sur les **16 384 pixels** du patch → une seule `C` de forme `(4, 4)` `complex128` |
| A10b | formes fermées | `QueganEstimator.solve`, `QueganEstimator.estimate` | `Δ`, `u, v, w, z` (Babu, équation 13), `X`, `alpha` → un `CrosstalkParameters` par patch ; aucune validation : un patch dégénéré (constant, nul, canal éteint, HH et VV parfaitement corrélés) rend des `nan` ou des valeurs finies sans signification ; aucun `nan` ni échec sur les 18 440 patchs de test |
| A10c | correction | `CrosstalkEstimator.estimate_and_correct` → `correct(patch, theta)` → `matrix.apply_correction` → `sigma_calibration_matrix`, `apply_matrix` | `Sigma` `(4, 4)`, protégée contre `(uw−1)(vz−1) ≈ 0`, puis `patch_corrige = Sigma · s'` pour chaque pixel ; rend `(patch_corrige, theta_initial)` |
| A10d | ré-estimation et réciprocité | `estimate(patch_corrige)`, `metrics.reciprocity` | `ResidualReport` : `theta_initial`, `theta` résiduel, réciprocité avant et après ; `evaluate_patches` le reçoit sous le nom `residual_report_patch` |
| A11 | synthèse | `run_baselines.aggregate`, `summarize` | la médiane des 18 440 lignes → tableau affiché ; rien n'est écrit sur disque |

Le solveur exact (`estimators/exact.py`) est enregistré sous le nom `exact` mais
absent du YAML : ajouter `- name: exact` à `baselines.estimators` pour l'évaluer. Il
résout par moindres carrés, patch par patch, les 10 contraintes de symétrie de
réflexion et de réciprocité, en partant de la solution de Quegan.

### Banc d'essai — `python src/train_mlp.py`, `python src/train_resnet.py`

Les deux scripts suivent les mêmes étapes et s'appuient sur les briques de `cvnn`,
sans `BaseExperiment`. Ils ne diffèrent que par leur YAML et par la forme d'entrée
donnée à `check_pipeline`.

| # | étape | code | entrée → sortie |
|---|---|---|---|
| R1 | lire la configuration | `cvnn.config.load_config`, `experiment.resolve_trainpath`, `from_repo_root` | YAML → `dict`, chemins lus depuis la racine du dépôt |
| R2 | découper | `set_seed(seed)`, `validate_and_correct_config`, `get_dataloaders` | la même partition que la chaîne A : 147 535 / 18 440 / 18 440 patchs |
| R3 | **injecter** | `MLPInjectionTransform` ou `ResNetInjectionTransform` → `CrosstalkInjector.inject` | un patch avec un zéro ou une valeur non finie (bord de fauchée) est écarté : entrée nulle, cible `nan`. Sinon : `theta_k = CrosstalkSampler.generate(patch_rng(patch, seed_injection))`, `matrix.apply_distortion`, puis division par la racine de la trace de la covariance |
| R4 | contrôler | `check_pipeline` | refuse un lot dont la forme n'est pas celle de la transformation attendue |
| R5 | mettre en cache | `cache.cached_loaders` → `build_cache` | une passe sur la pile, puis lecture par memmap dans `cache/<transformation>_<empreinte>/`. À chaque chargement, 8 patchs par partition sont recalculés et comparés au cache |
| R6 | préparer la sortie | `setup_wandb`, puis écriture de `config.yml` | `logs/<task>_<nom du run wandb, ou date et heure>/` |
| R7 | construire le réseau | `set_seed(seed_train)`, `cvnn.models.create_model`, `experiment.init_weights` | classe enregistrée sous `MLP` ou `ResNet` dans le registre de cvnn |
| R8 | **entraîner** | `CrosstalkLoss`, `cvnn.train.train_model` | le meilleur modèle en validation est gardé dans `best_model.pt` ; les courbes vont dans `historique.yaml` |
| R9 | **évaluer** | `experiment.evaluate` | `best_model.pt` sur le test, une seule fois → `evaluation.yaml` |

**Les deux réseaux.** Ils ont la même cible, `(u, v, w, z, alpha)` de `theta_k`, de
forme `(B, 5)`. Pour un même patch, la covariance de l'entrée du ResNet est l'entrée
du MLP : le MLP est à information égale avec Quegan, et le ResNet voit en plus les
pixels.

| | MLP | ResNet |
|---|---|---|
| module | [mlp_model.py](src/crosstalk/deep_learning/mlp_model.py) | [resnet_model.py](src/crosstalk/deep_learning/resnet_model.py) : le ResNet de cvnn, avec une tête de régression |
| transformation | `crosstalk_injection_mlp` | `crosstalk_injection_resnet` |
| entrée | les 10 entrées distinctes de `C`, `(B, 10, 1, 1)` | le patch injecté, `(B, 4, 128, 128)` |
| architecture | blocs linéaires de 128, 256 et 512 neurones, puis une couche de sortie | une convolution à 32 canaux, 5 blocs résiduels (1 024 canaux, 4 × 4 px en sortie), puis une tête de régression |
| paramètres réels | 342 538 | 41 262 282 |
| cache complet | environ 23 Mo | environ 97 Go (calcul : 184 415 patchs × 4 × 128² × 8 octets) |

Réglages communs : CReLU, BatchNorm, `kaiming_normal`, AdamW (taux 0,002,
`weight_decay` 0,05, écrêtage à 1,0), décroissance en cosinus, 50 époques, lots
de 256.

**La perte.** `CrosstalkLoss` somme `|ŷ − y|²` sur les 5 composantes et la moyenne
sur les patchs valides, sans normalisation. Les patchs écartés sont retirés avant la
soustraction. La docstring de
[loss.py](src/crosstalk/deep_learning/loss.py) donne le poids relatif de `alpha`.

**La loi d'injection.** Ses valeurs sont justifiées, avec leurs sources, dans
[crosstalk_sampler.py](src/crosstalk/crosstalk_sampler.py).

| champ | valeur | sens |
|---|---|---|
| `crosstalk_db` | [−35, −10] | enveloppe de `\|u\|, \|v\|, \|w\|, \|z\|`, en dB |
| `term_spread_db` | 4 | écart de chaque terme à leur niveau commun, en dB |
| `alpha_module_db` | [−3, 3] | `\|alpha\|`, en dB |
| `alpha_phase_deg` | [−30, 30] | `arg(alpha)`, en degrés |
| `no_injection_rate` | 0 | part de patchs laissés sans injection |

Les phases de `u, v, w, z` sont uniformes, par choix de méthode.

**Les trois graines.**

| clé | ce qu'elle fixe | change le cache ? |
|---|---|---|
| `seed` | le découpage train / valid / test | oui |
| `seed_injection` | les `theta_k` de chaque patch | oui |
| `seed_train` | l'initialisation des poids et l'ordre des lots | non |

Faire varier `seed_train` seule mesure la dispersion d'un entraînement à l'autre,
sur la même partition de test et avec les mêmes `theta_k`.

**Le cache.** L'empreinte du dossier couvre la pile, la taille des patchs, le
découpage, `max_train_samples`, les transformations et les composantes de la cible.
Changer le réseau, l'optimiseur, `nepochs` ou `seed_train` réutilise donc le même
cache. Une modification du code d'injection qui change les couples contrôlés arrête
le chargement : il faut alors supprimer le dossier pour le reconstruire.

### Lire les résultats du banc d'essai

Chaque run écrit dans `logs/<task>_<run>/` : `config.yml`, `historique.yaml`,
`evaluation.yaml` et `best_model.pt`, ce dernier ignoré par git.

| clé de `evaluation.yaml` | contenu |
|---|---|
| `-35/-30` … `-15/-10` | par tranche de crosstalk injecté : médiane, en dB, de `10 log10` de la moyenne de `\|ŷ − y\|²` sur `u, v, w, z` |
| `aucune` | la même chose pour les patchs sans injection : `nan` tant que `no_injection_rate` vaut 0 |
| `n_<tranche>` | effectif de la tranche |
| `alpha_<tranche>` | par tranche : médiane, en dB, de `10 log10 \|alpha estimé − alpha\|²` |
| `alpha` | médiane de `\|alpha estimé − alpha\|` sur tous les patchs valides |
| `patchs` | nombre de patchs valides : 17 720 sur 18 440 à `seed` 16 |

Le niveau d'un patch, qui fixe sa tranche, vaut `20 log10` de la moyenne quadratique
de `|u|, |v|, |w|, |z|` de sa cible.

Avant de comparer deux runs :

- vérifier leurs `config.yml` : `logs/` mêle plusieurs graines de découpage, tailles
  de patch et lois d'injection ;
- `valid_alpha` dans `historique.yaml` est une moyenne de `|alpha estimé − alpha|²`,
  alors que `alpha` dans `evaluation.yaml` est une médiane de `|alpha estimé − alpha|` ;
- seul `sage-frost-60` porte les clés `alpha` de l'évaluation actuelle. Les 65 autres
  runs ont une clé `log_alpha`, et la plupart ont été entraînés sur `log alpha` avec
  une perte normalisée : cette clé et leurs valeurs de perte ne se comparent pas à
  celles d'aujourd'hui.

### Ordre de lecture recommandé

Des fondations vers les points d'entrée : chaque fichier ne dépend que de ceux qui
le précèdent.

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
| 10 | [src/crosstalk/estimators/exact.py](src/crosstalk/estimators/exact.py) | les 10 contraintes, le départ depuis Quegan, les `nan` en cas d'échec |
| 11 | [src/crosstalk/metrics.py](src/crosstalk/metrics.py) | le résiduel est **circulaire**, la réciprocité ne l'est pas |
| 12 | [src/patchs.py](src/patchs.py) | patchs de test servis par `cvnn`, lots dépliés en patchs |
| 13 | [src/run_baselines.py](src/run_baselines.py) | assemblage de la chaîne A |
| 14 | [src/crosstalk/crosstalk_sampler.py](src/crosstalk/crosstalk_sampler.py) | la loi de `theta_k`, ses sources ; `theta_k` n'est pas le crosstalk total |
| 15 | [src/crosstalk/patch_rng.py](src/crosstalk/patch_rng.py) | même patch, même `theta_k`, quel que soit le processus ou l'époque |
| 16 | [src/crosstalk/deep_learning/crosstalk_injection.py](src/crosstalk/deep_learning/crosstalk_injection.py) | l'injection commune, les deux entrées, `check_pipeline` |
| 17 | [src/crosstalk/deep_learning/loss.py](src/crosstalk/deep_learning/loss.py) | patchs écartés retirés avant la soustraction, aucune normalisation |
| 18 | [src/crosstalk/deep_learning/cache.py](src/crosstalk/deep_learning/cache.py) | ce qui entre dans l'empreinte, le contrôle à chaque chargement |
| 19 | [src/crosstalk/deep_learning/mlp_model.py](src/crosstalk/deep_learning/mlp_model.py), puis [resnet_model.py](src/crosstalk/deep_learning/resnet_model.py) | sortie sans activation, multipliée par `ECHELLE_SORTIE` |
| 20 | [src/crosstalk/deep_learning/experiment.py](src/crosstalk/deep_learning/experiment.py) | `init_weights`, `TRANCHES_DB`, `evaluate` |
| 21 | [configs/cvnn_mlp.yaml](configs/cvnn_mlp.yaml), [configs/cvnn_resnet.yaml](configs/cvnn_resnet.yaml) | même découpage et même loi dans les deux |
| 22 | [src/train_mlp.py](src/train_mlp.py), [src/train_resnet.py](src/train_resnet.py) | assemblage du banc d'essai ; moment où chaque graine est posée |
| 23 | [tests/](tests/), dont l'outil [tests/synthetic.py](tests/synthetic.py) | hors chaîne : ne servent qu'aux tests, à `theta` connu |

### Points à garder en tête

- **L'ordre des canaux est implicite** : il naît en A6, de l'ordre alphabétique du
  `dict` rendu par torchcvnn, et tout le reste en dépend. Un test verrouille
  `POL_ORDER == sorted(POL_ORDER)`.
- **Les deux chaînes n'ont pas la même normalisation** : scalaire global en A, trace
  de chaque patch sur le banc d'essai. C'est sans effet sur `theta`, qui ne dépend
  pas de l'échelle du patch.
- **`theta_k` n'est pas le crosstalk total.** Le patch injecté vaut
  `D(theta_k) D(theta_0) S`, et `D(theta_k) D(theta_0) = g D(theta_tot) K`, où
  `theta_tot` n'est pas `theta_0 + theta_k`. Un réseau estime donc un écart au
  crosstalk typique des données, alors que Quegan et le solveur exact visent
  `theta_tot` ; il ne calibre pas les données à lui seul.
- **cvnn peut ignorer une transformation en silence** : s'il ne peut pas la
  construire, il émet un simple avertissement et le réseau recevrait des patchs
  bruts. `check_pipeline` sert de parade.
- **`patch_stride` doit rester égal à `patch_size`** : cvnn groupe les patchs par
  emplacement exact, et des patchs qui se recouvrent passeraient d'une partition à
  l'autre sans aucune erreur.

## État du dépôt au 2026-11-05

**Ce qui est en place**

- Les trois YAML donnent la même partition (`seed` 16, patchs de 128 px, aucun
  emplacement partagé entre l'entraînement et le test). La chaîne A évalue toutefois
  des patchs bruts, et le banc d'essai des patchs injectés.
- `cvnn_mlp.yaml` et `cvnn_resnet.yaml` ont le même découpage et la même loi, ce que
  vérifie un test : le MLP et le ResNet voient les mêmes `theta_k`.
- Le run `mlp_crosstalk_injection_sage-frost-60` (`seed_train` 3) correspond au
  `cvnn_mlp.yaml` actuel.

**Ce qui manque pour la comparaison Quegan / CVNN**

- Aucun script n'évalue Quegan et le solveur exact sur les patchs injectés des
  réseaux. La composition `D(theta_k) D(theta_0) = g D(theta_tot) K`, nécessaire pour
  les juger sur `theta_k`, n'est pas codée dans `src/`.
- `evaluate` ne rend que des agrégats, aucune erreur par patch.
- Le ResNet n'a aucun run à la configuration actuelle : les 8 de `logs/` sont en
  64 px, avec une autre graine de découpage.

**Tests.** `pytest tests` donne 437 réussites et 6 échecs. Les 6 échecs viennent de
tests qui attendent encore d'anciennes valeurs des YAML, pas d'un défaut du code
mesuré : patchs de 64 px, `seed_injection` 0, anciens nombres de paramètres. Les tests
qui lisent la pile sont ignorés si elle est absente.

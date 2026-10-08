# Audit de la chaîne A — `python src/run_baselines.py`

Ce guide suit la chaîne A maillon par maillon, du YAML au CSV, avec pour chaque
maillon son rôle informatique (entrées, sorties, formes, types), son rôle
mathématique, et ce qu'il faut vérifier en le relisant. Les fonctions sont
citées par leur nom plutôt que par leur numéro de ligne, qui change à chaque
modification du code.

Les chiffres correspondent à la configuration livrée : patchs de 64 px,
`seed: 38`, `test_ratio: 0.1`.

**Ce que fait la chaîne.** Pour chacun des 73 765 patchs de test : estimer
`theta = (u, v, w, z, alpha)` par Quegan, corriger le patch, ré-estimer, mesurer
la réciprocité HV/VH, et écrire une ligne dans `experiments/baselines.csv`.

```
configs/data_uavsar.yaml
   │
src/run_baselines.py ─ main
   ├─ 1. configuration ...... chemin_de_configuration, lire_configuration
   ├─ 2. méthodes ........... build_estimators → crosstalk/estimators/registry.py → QueganEstimator
   ├─ 3. données ............ build_dataloader → cvnn/src/cvnn/data.py (get_dataloaders)
   │                             ├─ torchcvnn SLCDataset / SLCFile   (lecture des .slc)
   │                             ├─ _groupes_par_emplacement          (découpage train/valid/test)
   │                             └─ cvnn/transform_registry.py        (PolSAR → tenseur → normalisation)
   ├─ 4. patchs + positions . iter_patches, positions_de_test, position_du_patch
   ├─ 5. boucle ............. evaluate_patches
   │     └─ crosstalk/metrics.py : residual_crosstalk
   │          ├─ estimators/base.py : estimate_and_correct
   │          │    ├─ estimators/quegan.py : estimate → covariance.py → _solve
   │          │    └─ physics.py : inverse_distortion_matrix, apply_matrix
   │          ├─ re-estimate sur le patch corrigé
   │          └─ reciprocity avant / après
   └─ 6. sorties ............ ecrire_resultats (CSV), summarize (médianes)
```

**Conventions utilisées partout.** Ordre des canaux `[HH, HV, VH, VV]`. Les
indices 1 à 4 des formules correspondent aux indices 0 à 3 du code.
`C_ij = <S'_i · S'_j*>`, la conjugaison portant sur le second indice.

---

## Étape 1 — Configuration

**Fichiers.** [configs/data_uavsar.yaml](../configs/data_uavsar.yaml) ;
[src/run_baselines.py](../src/run_baselines.py) : `main`,
`chemin_de_configuration`, `lire_configuration`.

**Informatique.**
- `chemin_de_configuration` résout `--config` dans cet ordre : chemin absolu,
  puis dossier courant, puis racine du dépôt.
- `lire_configuration` charge le YAML en `dict` et ancre `trainpath` à la racine
  du dépôt. Sans cela, cvnn chercherait les `.slc` depuis le dossier courant.

**Mathématique.** Rien.

**À auditer.** `seed: 38`, `patch_size: 64`, `test_ratio: 0.1`, les quatre
`cov_complex_value` identiques (voir l'étape 5), et `baselines.estimators`, qui
ne contient que `{name: quegan}`.

## Étape 2 — Instancier la méthode

**Fichiers.** `run_baselines.build_estimators` →
[estimators/registry.py](../src/crosstalk/estimators/registry.py)
(`create_from_config`, `create_estimator`) →
[estimators/quegan.py](../src/crosstalk/estimators/quegan.py).

**Informatique.**
- Importer le paquet `estimators` exécute `@register_estimator` sur
  `QueganEstimator`, ce qui remplit `_ESTIMATORS["quegan"]`.
- `create_from_config({"name": "quegan"})` appelle `QueganEstimator()` avec ses
  seuils par défaut : `tol = 1e-8`, `delta_tol = x_tol = 1e-6`.
- `build_estimators` refuse un nom déclaré deux fois.

**Mathématique.** Rien.

## Étape 3 — Découvrir les fichiers, sans rien lire

**Fichiers.** `run_baselines.build_dataloader` → cvnn `set_seed(38)`, puis
`validate_and_correct_config`, puis `get_dataloaders` → `_create_dataset` →
torchcvnn `SLCDataset`, puis `SLCFile`.

**Informatique.**
- `set_seed` fixe `random`, `numpy` et `torch`. C'est `random` qui pilotera le
  mélange de l'étape 4.
- `validate_and_correct_config` lit l'entrée `UavsarSLC` de
  `cvnn/src/cvnn/dataset_registry.py` et ajoute `type: polsar`,
  `num_channels: 4` et `has_labels: False`.
- `SLCDataset` liste `*.slc` avec `glob`, crée un `SLCFile` par fichier et les
  regroupe par acquisition (`key` = site, trace, vol, date…), puis par
  polarisation.
- Chaque `SLCFile` lit son `.ann` pour connaître ses dimensions, ici
  61 349 × 9 874, et précalcule la grille :
  - lignes de patchs : `(61 349 − 64) // 64 + 1 = 958` ;
  - colonnes de patchs : `(9 874 − 64) // 64 + 1 = 154` ;
  - soit **147 532 patchs par acquisition**, 737 660 au total.

**Mathématique.** L'index global `i` d'un patch se décompose en
`i = Σ (comptes des acquisitions précédentes) + k`, puis
`k → (ligne, colonne) = ((k // 154) × 64, (k % 154) × 64)`.

**À auditer.** L'ordre des acquisitions est celui de `glob`, donc **pas l'ordre
chronologique**. C'est pourquoi le CSV désigne chaque acquisition par sa clé,
jamais par un rang.

## Étape 4 — Découper train / valid / test sans fuite

**Fichier.** `cvnn/src/cvnn/data.py` : `_groupes_par_emplacement`, puis le bloc
« Découpage aléatoire » de `get_dataloaders`.

**Informatique.**
- La pile étant co-enregistrée, les 5 index `e, e + N, …, e + 4N` (avec
  `N = 147 532`) désignent le même bout de sol. Ils forment un groupe.
- `random.shuffle(groupes)` mélange les **emplacements**, pas les patchs, puis
  les aplatit.
- Les tailles sont arrondies au multiple de 5 inférieur :
  `int(0,1 × 737 660) = 73 766`, arrondi à 73 765.
- Résultat : 590 130 patchs en train, 73 765 en valid et 73 765 en test, le
  test en fin de liste.

**Pourquoi.** Le crosstalk d'un emplacement est corrélé d'une date à l'autre.
Mélanger patch par patch plaçait 99,9 % des emplacements de test aussi dans le
train : on aurait mesuré de la mémorisation du sol.

**À auditer.** La reproductibilité : même graine, même découpage
(`test_le_decoupage_est_rendu_reproductible`). Chaque emplacement de test doit
être présent avec ses 5 dates (vérifié dans
[notebooks/analyse_baselines.ipynb](../notebooks/analyse_baselines.ipynb), §1).

## Étape 5 — Transformer un patch en tenseur

**Fichier.** `cvnn/src/cvnn/transform_registry.py` :
`build_transform_pipeline`, qui enchaîne `PolSAR(out_channel=4)`, puis
`ToTensor(complex64)`, puis `GlobalScalarNormalize`.

**Informatique.**
- `SLCDataset.__getitem__` construit `{pol: patch}` dans l'ordre
  `sorted(keys)`, soit **HH, HV, VH, VV**, l'ordre alphabétique.
- `PolSAR` → `polsar_dict_to_array` empile les valeurs du dictionnaire dans cet
  ordre : tableau `(4, 64, 64)`.
- `ToTensor` le convertit en tenseur `complex64`.
- `GlobalScalarNormalize` calcule `x → (x − 0) / sqrt(0,013521 + 0,013521)`,
  soit `x / 0,16444`. Les 4 canaux sont divisés par **le même scalaire réel**.

**Mathématique.**
- Une homothétie réelle globale `g` donne `C → C / g²`.
- Les formes fermées sont des rapports homogènes de degré 0 en `C` : `Δ` et les
  numérateurs sont tous deux de degré 2.
- Donc `theta`, la réciprocité et le crosstalk en dB sont **invariants**.
- Une normalisation **par canal** reviendrait à une matrice diagonale `Λ` :
  `Λ·s' = (Λ·D)·s`, et les rapports de `D`, donc `theta`, seraient faussés.

**À auditer.**
- **L'ordre des canaux.** Il naît ici, de façon implicite, et tout le reste en
  dépend. Une erreur d'ordre ne lève rien : elle renomme `(u, v, w, z, alpha)`
  en `(z, w, v, u, 1/alpha)`. Le test `POL_ORDER == sorted(POL_ORDER)` le
  verrouille.
- **La normalisation.** Les quatre `cov_complex_value` du YAML doivent rester
  identiques, et `recompute_statistics` à `False`.

## Étape 6 — Lire les patchs de test

**Fichiers.** cvnn `DataLoader(Subset(base_eval_dataset, test_indices),
batch_size=16, shuffle=False, num_workers=8)` → torchcvnn
`SLCFile.__getitem__`.

**Informatique.**
- 4 611 lots.
- Pour chaque patch, `SLCFile` fait 64 lectures (`seek` puis `fromfile`), une
  par ligne, à l'octet `(ligne_début + r) × 9 874 × 8 + colonne_début × 8`. Le
  format est `complex64`, soit 8 octets par pixel.
- C'est l'essentiel des 4 minutes de la chaîne.

## Étape 7 — Associer chaque patch à sa position

**Fichiers.** `run_baselines.iter_patches`, `positions_de_test`,
`position_du_patch`.

**Informatique.**
- `iter_patches` refuse un chargeur mélangé : il exige un `SequentialSampler`.
- Il déplie chaque lot `(16, 4, 64, 64)` en patchs numpy `(4, 64, 64)`
  `complex64`.
- `positions_de_test` relit `Subset.indices`, et `position_du_patch` refait
  exactement le calcul de l'étape 3 : retrancher les comptes des acquisitions,
  puis `(k // 154) × 64` et `(k % 154) × 64`.
- `zip(..., strict=True)` lève une erreur si patchs et positions ne sont pas en
  même nombre.

**À auditer.** Cette position est la clé qui permettra d'apparier Quegan et le
CVNN patch par patch. Tests : `test_position_du_patch`,
`test_un_chargeur_melange_est_refuse`,
`test_un_chargeur_qui_perd_des_patchs_est_refuse`.

## Étape 8 — La boucle d'évaluation

**Fichier.** `run_baselines.evaluate_patches`.

**Informatique.**
- Pour chaque `(position, patch)` et chaque méthode, la boucle appelle
  `residual_crosstalk`.
- Une `CrosstalkError` (covariance mal conditionnée ou `D` singulière) donne le
  statut `ill_conditioned` sans arrêter la boucle.
- Cette fonction ne connaît ni fichiers ni dataloader, ce qui la rend testable
  sur des patchs synthétiques.

## Étape 9 — Estimer, corriger, ré-estimer (le cœur mathématique)

**Fichier pilote.** [metrics.py](../src/crosstalk/metrics.py) :
`residual_crosstalk(patch, estimateur)` :

```
corrigé, θ_initial = estimateur.estimate_and_correct(patch)      # 9a à 9e
θ_résiduel        = estimateur.estimate(corrigé)                 # 9a à 9c à nouveau
réciprocité avant = reciprocity(patch) ; après = reciprocity(corrigé)   # 9f
```

### 9a. Le modèle physique — [physics.py](../src/crosstalk/physics.py), `distortion_matrix`

`s' = D·s`, avec `D` égale à l'équation 13 de Fore **réindexée** dans l'ordre
`[HH, HV, VH, VV]` :

```
        |  1      v/√α     w√α     vw  |        s'_HH = s_HH + (v/√α) s_HV + w√α s_VH + vw s_VV
  D  =  |  z      1/√α     wz√α    w   |        s'_HV = z s_HH + ...
        |  u      uv/√α    √α      v   |        s'_VH = u s_HH + ...
        |  uz     u/√α     z√α     1   |        s'_VV = uz s_HH + ...
```

`u` est la fuite de HH vers VH, `z` celle de HH vers HV, `v` celle de VV vers
VH, et `w` celle de VV vers HV. `alpha` est le déséquilibre entre HV et VH.

**À auditer.** `D` et `Sigma` sont comparées terme à terme à Fore réindexé
(`test_d_est_la_matrice_de_fore_reindexee`), et `Sigma · D = I`. Les équations
10 et 12 de Babu sont fausses et ne sont pas utilisées (README, « Les deux
articles de référence »).

### 9b. La covariance — [covariance.py](../src/crosstalk/covariance.py), `polarimetric_covariance`

**Informatique.** Le patch `(4, 64, 64)` est passé en `complex128` et remis en
forme `(4, 4 096)` ; `C = V @ Vᴴ / 4 096` donne une matrice `(4, 4)`.

**Mathématique.** `C = (1/N) Σₙ kₙ kₙᴴ`, avec
`kₙ = [S'_HH, S'_HV, S'_VH, S'_VV]ᵀ` le vecteur du pixel `n`, soit
`C_ij = <S'_i S'_j*>`. `C` est hermitienne et semi-définie positive, et
`C_s' = D·C_s·Dᴴ`.

**À auditer.**
- **La conjugaison** porte sur le second indice
  (`test_la_convention_de_conjugaison_est_bien_c_ij_egale_oi_conj_oj`). Elle
  fixe les conjugués dans les formules de `alpha`.
- **Le lien `C_s' = D C_s Dᴴ`** est vérifié par
  `test_la_distorsion_transporte_la_covariance`.

### 9c. Les formes fermées — `QueganEstimator._solve`

**Hypothèses de Quegan sur le signal vrai.**
- Symétrie de réflexion : `<S_HH S_X*> = <S_VV S_X*> = 0`.
- Réciprocité : `S_HV = S_VH = S_X`.

**Conséquence.** Toute corrélation entre un canal co-polaire et un canal croisé
mesurée sur `s'` vient du crosstalk. Au premier ordre :

```
| C11  C41 | |u|   |C31|         | C11  C41 | |z|   |C21|
| C14  C44 | |v| = |C34|         | C14  C44 | |w| = |C24|
```

La règle de Cramer donne l'équation 13 de Babu, mot pour mot dans le code :

```
Δ = C11·C44 − |C14|²
u = (C44·C31 − C41·C34)/Δ        v = (C11·C34 − C31·C14)/Δ
z = (C44·C21 − C41·C24)/Δ        w = (C11·C24 − C21·C14)/Δ
```

**Le déséquilibre `alpha`**, d'après Babu 15 à 18 réindexées et Fore 25-26 :

```
X  = C23 − z·C13 − w·C43                      ≈ e^{−i arg α}·P_X  (une fois le crosstalk retiré)
α₁ = (C33 − u·C13 − v·C43) / X                ≈ α   (lu sur la puissance de VH)
α₂ = X* / (C22 − z*·C21 − w*·C24)             ≈ α   (lu sur la puissance de HV)
|α| = ( |α₁α₂| − 1 + √((|α₁α₂| − 1)² + 4|α₂|²) ) / (2|α₂|)       arg α = arg α₁
```

La formule du module concilie les deux estimations : `|alpha|` est l'unique
racine positive de `x − 1/x = |α₁| − 1/|α₂|`. Ce choix est verrouillé par
`test_l_estimateur_de_alpha_resout_l_identite_attendue`. C'est ici que le texte
de Fore porte la faute `4 + |α₂|²`, et que Babu omet la phase.

**Ce que la méthode néglige.** Les termes en `P_X·theta`. Il en résulte un biais
relatif proportionnel à `beta = P_X / sigma_co`, d'environ 30 % au `beta` médian
(README, « Le biais propre de Quegan »). Verrouillé par
`test_le_biais_est_le_terme_croise_neglige_par_les_formes_fermees`.

**À auditer.** Chaque indice, contre la table « Correspondance équation ↔ code »
du README. Ne pas « corriger » `C31` en `C21` : c'est la réindexation. Écrire la
formule de Fore non réindexée fait échouer 7 des 20 tests de `test_quegan.py`.

### 9d. Les garde-fous — `QueganEstimator.estimate`

Trois rapports sans dimension, invariants d'échelle :

| rapport | sens | doit être |
|---|---|---|
| `\|Δ\| / (C11·C44)` | `= 1 − \|ρ_HHVV\|²` | fini et > 1e-6 |
| `\|X\| / √(C22·C33)` | cohérence HV/VH une fois le crosstalk retiré | fini et > 1e-6 |
| `\|C22 − z*C21 − w*C24\| / C22` | dénominateur de `α₂` | fini et > 1e-6 |

Si l'un échoue, `IllConditionedCovarianceError` est levée et nomme le critère le
plus dégradé, un `nan` comptant pour −∞. Le cas typique est un canal éteint ou
un patch constant. Sur la pile, aucun patch n'échoue.

À la sortie, `CrosstalkParameters.from_vector`
([parameters.py](../src/crosstalk/parameters.py)) range le résultat dans un
objet immuable `(u, v, w, z, alpha)` et refuse `alpha = 0`.

### 9e. La correction — `CrosstalkEstimator.correct` → `physics.apply_correction`

**Mathématique.** `ŝ = Sigma·s'`, avec `Sigma = D⁻¹` sous forme analytique
(Fore, équation 15, réindexée) :

```
                          |  1        −v        −w       vw    |
Sigma = 1/((uw−1)(vz−1)) · | −z√α      √α       wz√α    −w√α   |
                          | −u/√α     uv/√α    1/√α    −v/√α  |
                          |  uz       −u        −z       1     |
```

Le bruit n'est pas retranché (`N` est inconnu) : la correction l'amplifie
légèrement.

**Informatique.**
- Si `|(uw−1)(vz−1)| < 1e-8`, `SingularDmatrixError` est levée. Sous 0,5, un
  avertissement est émis.
- `apply_matrix` contracte la matrice avec l'axe des canaux
  (`einsum "ij,...j->...i"`), calcule en `complex128` et rend du `complex64`.

### 9f. Les mesures — `metrics.reciprocity`, `ResidualReport`

- **Crosstalk en dB :** `max(20·log10 |u|, |v|, |w|, |z|)`
  (`CrosstalkParameters.max_crosstalk_db`). On l'obtient pour `θ_initial`
  (`initial_db`) et pour `θ_résiduel` (`residual_db`). Leur différence donne
  `reduction_db`.
- **Réciprocité :** `|<HV·VH*>| / √(<|HV|²>·<|VH|²>)`, avant et après
  correction. C'est le seul juge indépendant de Quegan, mais il est aveugle à
  `alpha` et à la direction `u = αz, v = αw`.
- **Le résiduel est jugé par l'estimateur lui-même.** Le notebook d'analyse
  montre qu'il suit le crosstalk initial environ 10 dB en dessous : il reflète
  surtout l'erreur relative de Quegan. Il ne prouve pas que la correction est
  juste.

## Étape 10 — Les sorties

**Fichiers.** `run_baselines.ecrire_resultats`, `summarize`.

**Informatique.**
- `ResidualReport.as_row` aplatit le rapport, avec `θ_initial` en parties réelle
  et imaginaire (`u_re`, `u_im`…).
- `csv.DictWriter` écrit les colonnes dans l'ordre de `COLONNES` ; un champ
  imprévu lève une erreur au lieu d'être perdu.
- Le CSV est écrasé à chaque lancement.
- `summarize` affiche la **médiane** des lignes `ok` pour chaque méthode.

---

## Ce que les tests verrouillent, maillon par maillon

| maillon | fichier de tests | ce qui est garanti |
|---|---|---|
| configuration, positions, CSV, découpage | `tests/test_run_baselines.py` | config livrée valide, position de chaque index, chargeur mélangé refusé, CSV relu à l'identique, découpage reproductible |
| registre, interface | `tests/test_registry.py`, `tests/test_estimator_base.py` | instanciation par nom, `correct` inverse exactement `D` |
| covariance | `tests/test_covariance.py` | convention `C_ij = <S'_i S'_j*>`, hermitienne et semi-définie positive, `C_s' = D C_s Dᴴ`, invariance d'échelle |
| formes fermées et `alpha` | `tests/test_quegan.py` | exactes sur la covariance théorique, structure du biais, identité de `alpha`, garde-fous, invariance d'échelle |
| modèle physique | `tests/test_physics.py` | `D` et `Sigma` identiques à Fore réindexé, `Sigma·D = I`, `POL_ORDER` trié |
| mesures | `tests/test_metrics.py` | réciprocité et ses angles morts, formule au réflecteur |

```bash
python -m pytest tests/ -q
```

## Les cinq points d'audit prioritaires

Chacun se tromperait **sans lever d'erreur** :

1. **L'ordre des canaux** (étape 5), implicite.
2. **La conjugaison** de `C` (étape 9b).
3. **La réindexation** par rapport à Fore (étape 9c).
4. **La normalisation**, qui doit rester globale (étape 5).
5. **L'association patch ↔ position** (étape 7).

Les autres points font échouer un test, ou lèvent une exception, s'ils sont
faux.

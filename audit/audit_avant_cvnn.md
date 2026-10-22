# Audit avant la partie CVNN — 2026-10-22

Question posée : le projet est-il prêt pour implémenter un encodeur CVNN qui
estime par régression les paramètres de crosstalk `(u, v, w, z, alpha)`, afin
de comparer son estimation à celle de Quegan, patch par patch ?

**Verdict.** Oui pour passer à la partie CVNN, pas encore pour écrire du code.
La baseline Quegan est solide et vérifiée, et l'architecture permet de brancher
un CVNN sur exactement les mêmes patchs. Une décision conditionne tout le
reste : d'où viennent les valeurs cibles de `theta` de la régression. Sur
données réelles, il n'en existe aucune.

## 1. Ce qui est prêt

| brique | état | ce qui le montre |
|---|---|---|
| Modèle physique D et Σ | ✔ | identiques au bit près aux équations 13 et 15 de Fore réindexées ; `Σ·D = I` à 1e-16 près |
| Quegan | ✔ | formes fermées exactes sur covariance analytique (tests) ; biais connu (~30 % du crosstalk) ; 0 échec sur les 73 765 patchs de test |
| Résultats de référence | ✔ | médianes −20,3 → −30,2 dB, réciprocité 0,767 → 0,788 ; réflecteurs −23,95 → −36,32 dB |
| Interface commune | ✔ | `CrosstalkEstimator.estimate(patch) → CrosstalkParameters`, registre, liste d'estimateurs du YAML : un estimateur `cvnn` enregistré est évalué par `run_baselines` sur les mêmes patchs, sans modifier le script |
| Données | ✔ | patchs 64×64 ; découpage groupé par emplacement (pas de fuite entre dates) ; normalisation par un scalaire commun, qui laisse `theta` inchangé |
| Outils synthétiques | ✔ en partie | `synthetic.py` (`C_Q`, Fore éq. 17), `apply_distortion`, `parameter_errors` |
| Environnement | ✔ | RTX 5090 visible par CUDA, torch 2.14, torchcvnn 0.9.9, convolution complexe sur GPU |
| Dépôt | ✔ | 177 tests passent ; tout est commité et poussé |

## 2. Décisions à prendre avant de coder

### D1. Source des valeurs cibles de `theta` (question centrale)

| option | valeurs cibles | comparaison possible | risque |
|---|---|---|---|
| A. Synthétique (`synthetic.py` + `apply_distortion`) | exactes | erreur par patch contre la vérité, pour les deux méthodes | écart au réel : signal gaussien, sans bruit, sans cibles brillantes |
| B. Semi-synthétique : patchs réels corrigés par Quegan puis re-distordus par un `theta` connu | `theta` connu + erreur résiduelle de Quegan (~−30 dB) | idem, sur textures réelles | cibles entachées du biais de Quegan |
| C. Auto-supervisé : la perte mesure les violations de réciprocité et de symétrie après correction | aucune | réel seulement | le CVNN optimise les hypothèses de Quegan/Ainsworth ; la réciprocité ne peut plus juger |
| D. Imiter Quegan | celles de Quegan | rien | circulaire : à exclure |

Recommandation : A d'abord (seul cadre où la comparaison patch par patch se fait
contre une vraie valeur ; le gain attendu du CVNN est de supprimer le biais au
premier ordre de Quegan), puis B pour tester la robustesse sur textures réelles.

### D2. Échelle commune synthétique / réel

Les patchs réels sont divisés par 0,16444 ; le générateur produit une puissance
d'environ 1. Les mettre à la même échelle, par un facteur commun aux 4 canaux
(ce qui laisse `theta` inchangé).

### D3. Distribution d'entraînement

- `theta` : modules de −40 à −10 dB, petites valeurs comprises (la
  ré-estimation après correction tombe vers −30 dB) ; `|α|` de 0,95 à 1,26,
  avec sa phase.
- Scène, autour des valeurs mesurées : `β` médian 0,119 ; `|ρ_HHVV|` ≈ 0,57 ;
  rapport d'amplitude VV/HH ≈ 0,73.
- Bruit : absent du générateur ; les canaux croisés sont ~3× plus faibles que
  HH en amplitude (RMS 0,078 contre 0,249).

### D4. Entrée du réseau

Le crosstalk se lit dans les statistiques d'ordre 2. Des convolutions complexes
sur les pixels doivent apprendre à former des produits `z_i·z_j*` puis des
rapports. Pilote à faire tôt : sur synthétique sans bruit, le CVNN doit au moins
égaler Quegan ; sinon, envisager une entrée de type covariance.

### D5. Sortie et perte

- 5 sorties complexes : ResNet de `cvnn` avec `projection_layer: null`
  (`"amplitude"`, la valeur par défaut, perd la phase).
- Échelles différentes (`u, v, w, z` ~0,1 ; `α` ~1,2) : normaliser les cibles
  ou pondérer la perte.
- Piège : dans `cvnn/src/cvnn/train.py` (l. 77-78), une `MSELoss` ou
  `ComplexMSELoss` est comparée aux **entrées** à l'entraînement, aux **cibles**
  en validation. La perte de régression doit être une classe dédiée.

## 3. À construire

`cvnn` fournit l'encodeur, la boucle d'entraînement et un système de tâches
extensible, mais aucune tâche de régression.

1. Tâche `regression` : `cvnn/projects/regression/`, plus une branche dans
   `model_utils.build_model_from_config` et dans `evaluate`.
2. Jeu de données avec valeurs cibles, enregistré dans `dataset_registry`
   (`UavsarSLC` : `has_labels: False`).
3. Perte de régression (D5).
4. `src/crosstalk/estimators/cvnn.py`, enregistré `cvnn`, qui rend un
   `CrosstalkParameters` par patch.
5. Évaluation synthétique appliquant `parameter_errors` aux deux méthodes.

## 4. Pour la comparaison finale (non bloquant pour démarrer)

- Comparaison patch par patch : les lignes ne portent plus l'identité du patch
  (positions retirées le 2026-10-08) et seules les médianes sortent. Ajouter
  l'indice du patch et écrire les lignes dans un CSV.
- Chaîne B : la bande (21 colonnes × 61 349 lignes) n'est pas lisible par un
  réseau entraîné sur 64×64. Géométrie commune à définir pour les deux
  méthodes ; trancher aussi l'exclusion des lignes du réflecteur (circularité,
  ~1,5 dB).
- Juges réels : le résiduel est circulaire pour chaque méthode ; la réciprocité
  n'est indépendante du CVNN que si sa perte ne l'utilise pas ; aucun juge réel
  ne voit `α`.
- Documentation : la justification de la normalisation n'est plus écrite dans
  le dépôt. Ancien README complet : `git show dd0ff88:README.md`.
- Mineur : `max_crosstalk_db` ignore un `nan` hors première position.

## 5. Prochaine étape

Trancher D1 et D4, puis valider une proposition d'architecture (fichiers,
classes, signatures, configuration YAML) avant d'écrire du code.

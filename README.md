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
- [`cvnn_framework`](https://github.com/QuentinGABOT/cvnn) — `get_dataloaders`
  découpe train / validation / test et construit les `DataLoader`.

Données attendues : `data/Rosamd_35012_04/`, pile UAVSAR *Stack SLC* de
5 acquisitions × 4 polarisations, 61 349 × 9 874 pixels `complex64` par fichier
(4,85 Go chacun, ~92 Go au total).

```python
import sys, yaml
sys.path.insert(0, "cvnn_framework/src")
from cvnn.data import get_dataloaders, validate_and_correct_config

cfg = validate_and_correct_config(yaml.safe_load(open("configs/data_uavsar.yaml")))
train_loader, valid_loader, test_loader = get_dataloaders(cfg, use_cuda=True)

batch = next(iter(train_loader))    # (B, 4, H, W) complex64, canaux [HH, HV, VH, VV]
```

Tout se règle dans [configs/data_uavsar.yaml](configs/data_uavsar.yaml) :
taille et pas des patchs, ratios de découpe, `batch_size`, `num_workers`,
transforms.

### Branchement dans cvnn_framework

`UavsarSLC` a été ajouté au framework comme n'importe quel autre dataset, en
deux endroits :

- `cvnn_framework/src/cvnn/dataset_registry.py` — entrée `UavsarSLC`
  (`type: polsar`, 4 canaux, sans étiquettes) ;
- `cvnn_framework/src/cvnn/data.py` — import de `SLCDataset`, branche
  `elif dataset_name == "UavsarSLC"` dans `_create_dataset`, et `UavsarSLC`
  ajouté à la liste des datasets à `patch_size`.

⚠️ `cvnn_framework/` est dans le `.gitignore` de ce dépôt : cette modification
n'y est pas suivie. La reporter dans le dépôt `cvnn` en amont, ou retirer
`*cvnn_framework` du `.gitignore`, pour ne pas la perdre.

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

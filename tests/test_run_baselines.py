"""Tests des fonctions pures du script d'orchestration.

Le parcours de la vraie pile n'est pas testé ici : il demande les 92 Go de la
pile UAVSAR. C'est précisément pourquoi `evaluate_patches` ne connaît ni
fichiers ni dataloader et se teste sur des patchs synthétiques. Le passage des
index aux positions se teste, lui, sur une pile factice qui imite les attributs
de `SLCDataset` et de `SLCFile`.
"""

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import apply_distortion
from src.crosstalk.synthetic import reflection_symmetric_scattering

RACINE = Path(__file__).resolve().parent.parent


def charge_script():
    chemin = RACINE / "src" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("run_baselines", chemin)
    module = importlib.util.module_from_spec(spec)
    # Inscrit avant exécution, comme le prévoit la recette d'importlib :
    # `@dataclass` y cherche le module pour résoudre les annotations différées.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rb = charge_script()

THETA = CrosstalkParameters(0.08 + 0.05j, -0.06 + 0.09j, 0.10 - 0.03j, -0.04 - 0.07j, 1.03)


def positionne(patchs):
    """Associe à chaque patch une position distincte, comme le fait `iter_patches`."""
    for rang, patch in enumerate(patchs):
        yield rb.PositionPatch("synthetique", 0, 256 * rang), patch


def patchs_synthetiques(nombre=3, taille=(128, 128)):
    for graine in range(nombre):
        signal = reflection_symmetric_scattering(
            taille, beta=0.01, rng=np.random.default_rng(graine)
        )
        yield apply_distortion(signal, THETA)


def estimateurs():
    return rb.build_estimators([{"name": "quegan"}])


# ------------------------------------------------------------- configuration


def test_la_configuration_livree_est_valide():
    """La config unique du dépôt doit se charger et s'instancier telle quelle.

    Les noms attendus sont déduits du fichier, pas recopiés : le test reste
    valable quand on ajoute ou retire un estimateur du YAML, et ne vérifie que
    le contrat — chaque entrée déclarée donne un estimateur, rangé sous son nom.
    """
    chemin = RACINE / "configs" / "data_uavsar.yaml"
    config = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    declarees = config["baselines"]["estimators"]
    construits = rb.build_estimators(declarees)

    assert construits, "aucun estimateur déclaré dans baselines.estimators"
    assert list(construits) == [entree["name"] for entree in declarees]


def test_la_cle_est_le_nom_de_la_methode():
    construits = rb.build_estimators([{"name": "quegan"}])
    assert list(construits) == ["quegan"]


def test_methode_declaree_deux_fois_refusee():
    """Deux entrées de même `name` s'écraseraient silencieusement dans le dict."""
    with pytest.raises(ValueError, match="deux fois"):
        rb.build_estimators([{"name": "quegan"}, {"name": "quegan"}])


def test_methode_inconnue_refusee():
    with pytest.raises(KeyError):
        rb.build_estimators([{"name": "ainsworth"}])


# ---------------------------------------------------------------- évaluation


def test_une_ligne_par_patch_et_par_methode():
    lignes = rb.evaluate_patches(positionne(patchs_synthetiques(3)), estimateurs())
    assert len(lignes) == 3
    assert all(ligne["name"] == "quegan" for ligne in lignes)
    assert all(ligne["status"] == "ok" for ligne in lignes)
    assert all(ligne["reduction_db"] > 0 for ligne in lignes)


def test_chaque_ligne_porte_la_position_de_son_patch():
    lignes = rb.evaluate_patches(positionne(patchs_synthetiques(3)), estimateurs())
    assert [(l["acquisition"], l["ligne"], l["colonne"]) for l in lignes] == [
        ("synthetique", 0, 0), ("synthetique", 0, 256), ("synthetique", 0, 512),
    ]


def test_un_patch_degenere_est_consigne_sans_interrompre():
    patchs = [
        np.ones((4, 32, 32), dtype=np.complex64),        # dégénéré
        next(iter(patchs_synthetiques(1, (64, 64)))),    # exploitable
    ]
    lignes = rb.evaluate_patches(positionne(patchs), estimateurs())
    assert [ligne["status"] for ligne in lignes] == ["ill_conditioned", "ok"]
    assert "initial_db" not in lignes[0]
    assert lignes[0]["colonne"] == 0, "un patch en échec garde sa position"


def test_la_ligne_et_le_fichier_coincident():
    """Aucun champ calculé puis jeté : tout ce que porte une ligne est écrit."""
    ligne = rb.evaluate_patches(positionne(patchs_synthetiques(1)), estimateurs())[0]
    assert set(ligne) == set(rb.COLONNES)


# ------------------------------------------------------------------ sorties


def test_synthese_liste_les_methodes():
    lignes = rb.evaluate_patches(positionne(patchs_synthetiques(2)), estimateurs())
    texte = rb.summarize(lignes)
    assert "quegan" in texte and "réduction dB" in texte


def test_synthese_signale_une_methode_entierement_en_echec():
    lignes = rb.evaluate_patches(positionne([np.ones((4, 16, 16), dtype=np.complex64)]), estimateurs())
    assert "1       1" in rb.summarize(lignes)


def test_le_fichier_se_relit_a_l_identique(tmp_path):
    patchs = [np.ones((4, 16, 16), dtype=np.complex64), next(iter(patchs_synthetiques(1)))]
    lignes = rb.evaluate_patches(positionne(patchs), estimateurs())
    chemin = tmp_path / "sous_dossier" / "baselines.csv"
    rb.ecrire_resultats(lignes, chemin)

    with chemin.open(encoding="utf-8", newline="") as fichier:
        lecteur = csv.DictReader(fichier)
        assert tuple(lecteur.fieldnames) == rb.COLONNES
        echec, ok = list(lecteur)

    assert echec["status"] == "ill_conditioned"
    assert echec["initial_db"] == "" and echec["u_re"] == ""
    assert (ok["acquisition"], int(ok["ligne"]), int(ok["colonne"])) == ("synthetique", 0, 256)
    for nom in ("u", "v", "w", "z", "alpha"):
        attendu = complex(lignes[1][f"{nom}_re"], lignes[1][f"{nom}_im"])
        assert complex(float(ok[f"{nom}_re"]), float(ok[f"{nom}_im"])) == attendu


def test_un_champ_hors_des_colonnes_fait_echouer_l_ecriture(tmp_path):
    with pytest.raises(ValueError):
        rb.ecrire_resultats([{"name": "quegan", "status": "ok", "inconnu": 1.0}], tmp_path / "x.csv")


# --------------------------------------------------- positions dans la pile


class PileFactice(torch.utils.data.Dataset):
    """Imite les attributs de `SLCDataset` et de `SLCFile` que lit le script.

    Deux acquisitions, rangées comme le ferait `glob` : la plus récente
    d'abord. Grille de 2 x 3 patchs ; chaque patch porte son index global.
    """

    def __init__(self):
        self.patch_counts = {"B_160610": 6, "A_160211": 6}
        fichier = SimpleNamespace(nsamples_per_cols=3, patch_stride=(256, 256))
        self.slc_polarizations = {cle: {"HH": fichier, "VV": fichier} for cle in self.patch_counts}

    def __len__(self):
        return 12

    def __getitem__(self, index):
        return np.full((4, 2, 2), index, dtype=np.complex64)


@pytest.mark.parametrize(
    "index, attendu",
    [
        (0, ("B_160610", 0, 0)),
        (2, ("B_160610", 0, 512)),
        (3, ("B_160610", 256, 0)),
        (6, ("A_160211", 0, 0)),
        (11, ("A_160211", 256, 512)),
    ],
)
def test_position_du_patch(index, attendu):
    comptes = PileFactice().patch_counts
    assert rb.position_du_patch(index, comptes, 3, (256, 256)) == rb.PositionPatch(*attendu)


def test_un_index_hors_de_la_pile_est_refuse():
    with pytest.raises(IndexError):
        rb.position_du_patch(12, PileFactice().patch_counts, 3, (256, 256))


def test_chaque_patch_recoit_la_position_de_son_index():
    """Le passage `Subset.indices` -> position, à travers un vrai `DataLoader` en lots."""
    sous_ensemble = torch.utils.data.Subset(PileFactice(), [7, 0, 11, 4])
    chargeur = torch.utils.data.DataLoader(sous_ensemble, batch_size=3, shuffle=False)
    couples = list(rb.iter_patches(chargeur))

    assert [int(patch[0, 0, 0].real) for _, patch in couples] == [7, 0, 11, 4]
    assert [position for position, _ in couples] == [
        rb.PositionPatch("A_160211", 0, 256),
        rb.PositionPatch("B_160610", 0, 0),
        rb.PositionPatch("A_160211", 256, 512),
        rb.PositionPatch("B_160610", 256, 256),
    ]


def test_un_chargeur_melange_est_refuse():
    sous_ensemble = torch.utils.data.Subset(PileFactice(), [7, 0, 11, 4])
    chargeur = torch.utils.data.DataLoader(sous_ensemble, batch_size=3, shuffle=True)
    with pytest.raises(ValueError, match="mélangé"):
        next(rb.iter_patches(chargeur))


def test_un_chargeur_qui_perd_des_patchs_est_refuse():
    """`drop_last` écarte le dernier lot incomplet : trois patchs pour quatre positions."""
    sous_ensemble = torch.utils.data.Subset(PileFactice(), [7, 0, 11, 4])
    chargeur = torch.utils.data.DataLoader(sous_ensemble, batch_size=3, shuffle=False, drop_last=True)
    with pytest.raises(ValueError):
        list(rb.iter_patches(chargeur))


def test_des_grilles_differentes_sont_refusees():
    pile = PileFactice()
    pile.slc_polarizations["A_160211"]["VV"] = SimpleNamespace(nsamples_per_cols=4, patch_stride=(256, 256))
    chargeur = torch.utils.data.DataLoader(torch.utils.data.Subset(pile, [0]), shuffle=False)
    with pytest.raises(ValueError, match="grilles"):
        rb.positions_de_test(chargeur)


def test_le_decoupage_est_rendu_reproductible():
    """`cvnn.data` découpe avec les générateurs globaux, jamais graines.

    Le script appelle `cvnn.utils.set_seed`, la même fonction que
    `BaseExperiment` utilise pour le réseau : les deux voient donc la même
    partition. Sans ce réamorçage, deux exécutions évaluent deux partitions de
    test différentes.
    """
    import torch
    from cvnn.utils import set_seed

    set_seed(1670)
    premier = torch.randperm(50).tolist()
    set_seed(1670)
    assert torch.randperm(50).tolist() == premier


# ------------------------------------------------------- chemins et configuration


def test_sans_option_la_configuration_par_defaut_est_celle_du_depot():
    assert rb.chemin_de_configuration(None) == RACINE / "configs" / "data_uavsar.yaml"


def test_un_chemin_relatif_se_resout_depuis_la_racine_hors_du_depot(tmp_path, monkeypatch):
    """Lancé depuis `src/` ou d'ailleurs, `--config configs/data_uavsar.yaml` doit aboutir."""
    monkeypatch.chdir(tmp_path)
    assert rb.chemin_de_configuration("configs/data_uavsar.yaml") == RACINE / "configs" / "data_uavsar.yaml"


def test_un_chemin_relatif_present_dans_le_dossier_courant_est_prioritaire(tmp_path, monkeypatch):
    (tmp_path / "ma_config.yaml").write_text("x: 1\n")
    monkeypatch.chdir(tmp_path)
    assert rb.chemin_de_configuration("ma_config.yaml") == Path("ma_config.yaml")


@pytest.mark.parametrize("absolu", [False, True])
def test_trainpath_est_ancre_a_la_racine_du_depot(tmp_path, monkeypatch, absolu):
    cible = "/donnees/ailleurs" if absolu else "data/Rosamd_35012_04"
    fichier = tmp_path / "c.yaml"
    fichier.write_text(f"data:\n  dataset:\n    trainpath: {cible}\n")
    monkeypatch.chdir(tmp_path)
    attendu = cible if absolu else str(RACINE / cible)
    assert rb.lire_configuration(fichier)["data"]["dataset"]["trainpath"] == attendu

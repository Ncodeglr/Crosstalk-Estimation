"""Tests des fonctions pures du script d'orchestration.

Le parcours du dataloader n'est pas testé ici : il demande les 92 Go de la pile
UAVSAR. C'est précisément pourquoi `evaluate_patches` ne connaît ni fichiers ni
dataloader et se teste sur des patchs synthétiques.
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.crosstalk.conventions import POL_ORDER_ONERA
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import apply_distortion
from src.crosstalk.synthetic import reflection_symmetric_scattering

RACINE = Path(__file__).resolve().parent.parent


def charge_script():
    chemin = RACINE / "src" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("run_baselines", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb = charge_script()

THETA = CrosstalkParameters(0.08 + 0.05j, -0.06 + 0.09j, 0.10 - 0.03j, -0.04 - 0.07j, 1.03)


def patchs_synthetiques(nombre=3, taille=(128, 128)):
    for graine in range(nombre):
        signal = reflection_symmetric_scattering(
            taille, beta=0.01, rng=np.random.default_rng(graine)
        )
        yield apply_distortion(signal, THETA)


def estimateurs():
    return rb.build_estimators(
        [
            {"label": "patch", "name": "quegan",
             "params": {"window": "full", "source_order": list(POL_ORDER_ONERA)}},
        ]
    )


# ------------------------------------------------------------- configuration


def test_la_configuration_livree_est_valide():
    """La config unique du dépôt doit se charger et s'instancier telle quelle."""
    config = rb.load_config(RACINE / "configs" / "data_uavsar.yaml")
    construits = rb.build_estimators(config["baselines"]["estimators"])
    assert [label for label, _ in construits] == [
        "quegan_patch",
        "quegan_5x5",
        "quegan_babu",
    ]
    assert construits[2][1].alpha_phase == "babu"


def test_etiquette_par_defaut_vaut_le_nom():
    construits = rb.build_estimators([{"name": "quegan"}])
    assert construits[0][0] == "quegan"


def test_etiquette_dupliquee_refusee():
    with pytest.raises(ValueError, match="dupliquée"):
        rb.build_estimators([{"name": "quegan"}, {"name": "quegan"}])


def test_methode_inconnue_refusee():
    with pytest.raises(KeyError):
        rb.build_estimators([{"name": "ainsworth"}])


# ---------------------------------------------------------------- évaluation


def test_une_ligne_par_patch_et_par_methode():
    lignes = rb.evaluate_patches(patchs_synthetiques(3), estimateurs())
    assert len(lignes) == 3
    assert [ligne["patch"] for ligne in lignes] == [0, 1, 2]
    assert all(ligne["status"] == "ok" for ligne in lignes)
    assert all(ligne["reduction_db"] > 0 for ligne in lignes)


def test_un_patch_degenere_est_consigne_sans_interrompre():
    patchs = [
        np.ones((4, 32, 32), dtype=np.complex64),        # dégénéré
        next(iter(patchs_synthetiques(1, (64, 64)))),    # exploitable
    ]
    lignes = rb.evaluate_patches(patchs, estimateurs())
    assert [ligne["status"] for ligne in lignes] == ["ill_conditioned", "ok"]
    assert "initial_db" not in lignes[0]


def test_les_colonnes_de_crosstalk_sont_remplies():
    ligne = rb.evaluate_patches(patchs_synthetiques(1), estimateurs())[0]
    for nom in ("u_db", "v_db", "w_db", "z_db"):
        assert ligne[nom] < 0.0


# ------------------------------------------------------------------ sorties


def test_synthese_liste_les_methodes():
    lignes = rb.evaluate_patches(patchs_synthetiques(2), estimateurs())
    texte = rb.summarize(lignes)
    assert "patch" in texte and "réduction dB" in texte


def test_synthese_signale_une_methode_entierement_en_echec():
    lignes = rb.evaluate_patches([np.ones((4, 16, 16), dtype=np.complex64)], estimateurs())
    assert "1       1" in rb.summarize(lignes)


def test_csv_relisible(tmp_path):
    lignes = rb.evaluate_patches(patchs_synthetiques(2), estimateurs())
    destination = tmp_path / "sous_dossier" / "resultats.csv"
    rb.write_csv(lignes, destination)
    with open(destination, encoding="utf-8") as fichier:
        relues = list(csv.DictReader(fichier))
    assert len(relues) == 2
    assert list(relues[0]) == rb.COLONNES
    assert relues[0]["status"] == "ok"


def test_csv_tolere_les_lignes_incompletes(tmp_path):
    lignes = rb.evaluate_patches([np.ones((4, 16, 16), dtype=np.complex64)], estimateurs())
    destination = tmp_path / "resultats.csv"
    rb.write_csv(lignes, destination)  # les colonnes absentes restent vides
    with open(destination, encoding="utf-8") as fichier:
        relue = next(csv.DictReader(fichier))
    assert relue["status"] == "ill_conditioned"
    assert relue["initial_db"] == ""


def test_le_decoupage_est_rendu_reproductible():
    """`cvnn.data` découpe avec les générateurs globaux, jamais graines.

    Sans ce réamorçage, deux exécutions évaluent deux partitions de test
    différentes — et rien ne garantit alors que la partition des baselines soit
    disjointe de celle sur laquelle le réseau a été entraîné.
    """
    import torch

    rb._seed_global_rngs(1670)
    premier = torch.randperm(50).tolist()
    rb._seed_global_rngs(1670)
    assert torch.randperm(50).tolist() == premier

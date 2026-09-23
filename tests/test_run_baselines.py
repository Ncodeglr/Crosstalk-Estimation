"""Tests des fonctions pures du script d'orchestration.

Le parcours du dataloader n'est pas testé ici : il demande les 92 Go de la pile
UAVSAR. C'est précisément pourquoi `evaluate_patches` ne connaît ni fichiers ni
dataloader et se teste sur des patchs synthétiques.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

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
    """La config unique du dépôt doit se charger et s'instancier telle quelle.

    Les étiquettes attendues sont déduites du fichier, pas recopiées : le test
    reste valable quand on ajoute ou retire un estimateur du YAML, et ne
    vérifie que le contrat — chaque entrée déclarée donne un estimateur, rangé
    sous son étiquette.
    """
    chemin = RACINE / "configs" / "data_uavsar.yaml"
    config = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    declarees = config["baselines"]["estimators"]
    construits = rb.build_estimators(declarees)

    assert construits, "aucun estimateur déclaré dans baselines.estimators"
    assert list(construits) == [e.get("label", e["name"]) for e in declarees]
    for entree, estimateur in zip(declarees, construits.values()):
        assert estimateur.name == entree["name"]


def test_etiquette_par_defaut_vaut_le_nom():
    construits = rb.build_estimators([{"name": "quegan"}])
    assert list(construits) == ["quegan"]


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

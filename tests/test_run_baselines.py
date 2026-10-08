"""Tests des fonctions pures du script d'orchestration.

Le parcours de la vraie pile n'est pas testé ici : il demande les 92 Go de la
pile UAVSAR. C'est précisément pourquoi `evaluate_patches` ne connaît ni
fichiers ni dataloader et se teste sur des patchs synthétiques.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
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
    chemin = rb.CONFIG
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
    lignes = rb.evaluate_patches(patchs_synthetiques(3), estimateurs())
    assert len(lignes) == 3
    assert all(ligne["name"] == "quegan" for ligne in lignes)
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


# ------------------------------------------------------------------ synthèse


def test_synthese_liste_les_methodes():
    lignes = rb.evaluate_patches(patchs_synthetiques(2), estimateurs())
    texte = rb.summarize(lignes)
    assert "quegan" in texte and "réduction dB" in texte


def test_synthese_signale_une_methode_entierement_en_echec():
    lignes = rb.evaluate_patches([np.ones((4, 16, 16), dtype=np.complex64)], estimateurs())
    assert "1       1" in rb.summarize(lignes)


# ------------------------------------------------------------- configuration


@pytest.mark.parametrize("absolu", [False, True])
def test_trainpath_est_ancre_a_la_racine_du_depot(tmp_path, monkeypatch, absolu):
    cible = "/donnees/ailleurs" if absolu else "data/Rosamd_35012_04"
    fichier = tmp_path / "c.yaml"
    fichier.write_text(f"data:\n  dataset:\n    trainpath: {cible}\n")
    monkeypatch.chdir(tmp_path)
    attendu = cible if absolu else str(RACINE / cible)
    assert rb.lire_YAML(fichier)["data"]["dataset"]["trainpath"] == attendu

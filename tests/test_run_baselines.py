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

from src.crosstalk.estimators.estimator import CrosstalkEstimator
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.matrix import apply_distortion
from synthetic import ScatteringParameters, sample_scattering

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
        signal = sample_scattering(ScatteringParameters(beta=0.01), taille, rng=np.random.default_rng(graine)
        )
        yield apply_distortion(signal, THETA)


def estimateurs():
    return rb.build_estimators([{"name": "quegan"}])


class EstimateurSingulier(CrosstalkEstimator):
    """Rend un jeu dont la matrice D est singulière : la correction échoue à chaque patch."""

    name = "singulier"

    def estimate(self, patch):
        self.check_patch(patch)
        return CrosstalkParameters(1.0, 0.0, 1.0, 0.0, 1.0)  # uw - 1 = 0


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


def test_un_echec_est_consigne_sans_interrompre():
    """Une méthode qui échoue n'arrête ni le parcours des patchs ni les autres méthodes."""
    methodes = {**estimateurs(), "singulier": EstimateurSingulier()}
    lignes = rb.evaluate_patches(patchs_synthetiques(2, (64, 64)), methodes)
    assert [(ligne["name"], ligne["status"]) for ligne in lignes] == [
        ("quegan", "ok"), ("singulier", "ill_conditioned"),
        ("quegan", "ok"), ("singulier", "ill_conditioned"),
    ]
    assert "initial_db" not in lignes[1]


# ------------------------------------------------------------------- bilan


def test_bilan_prend_la_mediane_des_patchs_exploitables():
    """Lignes écrites à la main : `aggregate` se teste sans estimateur."""
    lignes = [
        {"name": "quegan", "status": "ok", "reduction_db": 10.0},
        {"name": "quegan", "status": "ok", "reduction_db": 30.0},
        {"name": "quegan", "status": "ok", "reduction_db": 14.0},
        {"name": "quegan", "status": "ill_conditioned"},
    ]
    assert rb.aggregate(lignes) == {"quegan": {"patchs": 4, "echecs": 1, "reduction_db": 14.0}}


def test_bilan_separe_les_methodes_et_les_trie():
    lignes = [
        {"name": "b", "status": "ok", "alpha_abs": 1.0},
        {"name": "a", "status": "ok", "alpha_abs": 3.0},
        {"name": "b", "status": "ok", "alpha_abs": 5.0},
    ]
    bilan = rb.aggregate(lignes)
    assert list(bilan) == ["a", "b"]
    assert bilan["b"]["alpha_abs"] == 3.0


def test_bilan_laisse_voir_un_nan():
    """Un `nan` n'est pas écarté en silence : il se lit dans le tableau."""
    lignes = [{"name": "quegan", "status": "ok", "reciprocity_after": v} for v in (0.9, float("nan"))]
    assert np.isnan(rb.aggregate(lignes)["quegan"]["reciprocity_after"])


# ------------------------------------------------------------------ synthèse


def test_synthese_liste_les_methodes():
    lignes = rb.evaluate_patches(patchs_synthetiques(2), estimateurs())
    texte = rb.summarize(rb.aggregate(lignes))
    assert "quegan" in texte and "réduction dB" in texte


def test_synthese_signale_une_methode_entierement_en_echec():
    lignes = rb.evaluate_patches(patchs_synthetiques(1, (64, 64)), {"singulier": EstimateurSingulier()})
    bilan = rb.aggregate(lignes)
    assert bilan == {"singulier": {"patchs": 1, "echecs": 1}}
    ligne_singulier = rb.summarize(bilan).splitlines()[2]
    assert ligne_singulier.split() == ["singulier", "1", "1"] + ["-"] * 6


# ------------------------------------------------------------- configuration


@pytest.mark.parametrize("absolu", [False, True])
def test_trainpath_est_ancre_a_la_racine_du_depot(tmp_path, monkeypatch, absolu):
    cible = "/donnees/ailleurs" if absolu else "data/Rosamd_35012_04"
    fichier = tmp_path / "c.yaml"
    fichier.write_text(f"data:\n  dataset:\n    trainpath: {cible}\n")
    monkeypatch.chdir(tmp_path)
    attendu = cible if absolu else str(RACINE / cible)
    assert rb.lire_YAML(fichier)["data"]["dataset"]["trainpath"] == attendu

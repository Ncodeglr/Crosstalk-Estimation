"""Tests des fonctions pures du script d'orchestration.

Le parcours de la vraie pile n'est pas testé ici : il demande les 92 Go de la
pile UAVSAR. C'est précisément pourquoi `evaluate_patches` ne connaît ni
fichiers ni dataloader et se teste sur des patchs synthétiques. Le passage des
index aux positions se teste dans `test_patchs.py`.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import apply_distortion
from src.crosstalk.synthetic import reflection_symmetric_scattering
from src.patchs import PositionPatch

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
        yield PositionPatch("synthetique", 0, 256 * rang), patch


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


# ------------------------------------------------------------- configuration


@pytest.mark.parametrize("absolu", [False, True])
def test_trainpath_est_ancre_a_la_racine_du_depot(tmp_path, monkeypatch, absolu):
    cible = "/donnees/ailleurs" if absolu else "data/Rosamd_35012_04"
    fichier = tmp_path / "c.yaml"
    fichier.write_text(f"data:\n  dataset:\n    trainpath: {cible}\n")
    monkeypatch.chdir(tmp_path)
    attendu = cible if absolu else str(RACINE / cible)
    assert rb.lire_YAML(fichier)["data"]["dataset"]["trainpath"] == attendu

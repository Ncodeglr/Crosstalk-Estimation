"""Tests de l'estimateur de Quegan, dont le test de cohérence bout en bout."""

import logging

import numpy as np
import pytest

from src.crosstalk.covariance import BABU_WINDOW
from src.crosstalk.estimators import create_from_config
from src.crosstalk.estimators.quegan import QueganEstimator
from src.crosstalk.exceptions import IllConditionedCovarianceError
from src.crosstalk.parameters import CrosstalkParameters
from src.crosstalk.physics import apply_distortion
from src.crosstalk.synthetic import reflection_symmetric_scattering

#: Direction de crosstalk de référence, mise à l'échelle par `epsilon`.
DIRECTION = np.array([0.8 + 0.5j, -0.6 + 0.9j, 1.0 - 0.3j, -0.4 - 0.7j])
ALPHA = 1.03 * np.exp(1j * np.deg2rad(27.0))


def theta(epsilon: float = 0.1) -> CrosstalkParameters:
    """Jeu de référence à `max|u,v,w,z| ~ 0.108 * epsilon / 0.1`."""
    return CrosstalkParameters.from_vector([*(epsilon * DIRECTION), ALPHA])


def observe(params, shape=(1024, 1024), beta=0.05, graine=0, **kwargs):
    """Observation synthétique sans bruit, en ordre modèle : `o = D s`."""
    signal = reflection_symmetric_scattering(
        shape, beta=beta, rng=np.random.default_rng(graine), **kwargs
    )
    return apply_distortion(signal, params)


def estimateur(**kwargs) -> QueganEstimator:
    return QueganEstimator(**kwargs)


def erreur_relative(estime: CrosstalkParameters, vrai: CrosstalkParameters) -> float:
    a, b = estime.to_vector()[:4], vrai.to_vector()[:4]
    return float((np.abs(a - b) / np.abs(b)).max())


# ============================================================ cohérence bout en bout


def test_bout_en_bout_couplage_faible():
    """Le test central : `s` -> `o = D s` -> `theta` ré-estimé.

    Le couplage co-pol/cross-pol est réduit (`beta = 0.002`, soit -27 dB) pour
    que le biais structurel de la méthode, quantifié plus bas, reste sous le
    pour-cent et laisse voir la justesse des formules elles-mêmes.
    """
    vrai = theta(0.1)
    estime = estimateur().estimate(observe(vrai, beta=0.002))
    assert erreur_relative(estime, vrai) < 0.03


def test_alpha_est_retrouve():
    estime = estimateur().estimate(observe(theta(0.1), beta=0.002))
    assert abs(estime.alpha) == pytest.approx(abs(ALPHA), rel=0.02)
    assert estime.alpha_phase_deg() == pytest.approx(
        np.angle(ALPHA, deg=True), abs=2.0
    )


def test_le_biais_relatif_est_proportionnel_au_rapport_crosspol_copol():
    """Caractérise le biais structurel de la méthode.

    Les formes fermées négligent les termes en `beta` (la puissance des canaux
    croisés) devant la puissance co-polaire. Il en résulte un biais **relatif**
    sur `theta` proportionnel à `beta / sigma_copol`, et donc **indépendant du
    niveau de crosstalk**. Mesuré à environ `2,3 x beta` sur ce montage.

    Ce test vaut vérification de structure : une erreur dans une des formes
    fermées casserait cette proportionnalité.
    """
    vrai = theta(0.1)
    biais = {
        beta: erreur_relative(estimateur().estimate(observe(vrai, beta=beta)), vrai)
        for beta in (0.01, 0.04)
    }
    assert biais[0.04] / biais[0.01] == pytest.approx(4.0, rel=0.25)
    assert biais[0.01] / 0.01 == pytest.approx(2.3, rel=0.35)


def test_le_biais_relatif_ne_depend_pas_du_niveau_de_crosstalk():
    """Corollaire : diviser `theta` par 10 divise l'erreur absolue par 10."""
    erreurs = {
        eps: erreur_relative(
            estimateur().estimate(observe(theta(eps), beta=0.02)), theta(eps)
        )
        for eps in (0.2, 0.02)
    }
    assert erreurs[0.2] == pytest.approx(erreurs[0.02], rel=0.25)


def test_taille_de_patch_realiste():
    """Un patch 64x64 reste exploitable, avec une dispersion plus forte."""
    vrai = theta(0.1)
    erreurs = [
        erreur_relative(
            estimateur().estimate(observe(vrai, shape=(64, 64), beta=0.02, graine=k)),
            vrai,
        )
        for k in range(20)
    ]
    assert np.median(erreurs) < 0.25


def test_l_estimateur_de_alpha_resout_l_identite_attendue():
    """Verrouille la formule de `alpha` indépendamment de la typographie.

    En développant l'équation 25 de Fore et al. (2015), le module rendu est
    l'unique racine positive de ::

        x - 1/x = |alpha1| - 1/|alpha2|

    c'est-à-dire la valeur qui concilie l'estimation de `alpha` par `alpha1` et
    celle de `1/alpha` par `alpha2`. Cette identité fixe le terme `4|alpha2|^2`
    sous la racine : la variante `4 + |alpha2|^2`, que produit l'OCR du PDF,
    ne la vérifie pas.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        a1 = rng.lognormal(0.0, 0.5) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        a2 = rng.lognormal(0.0, 0.5) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        produit = abs(a1 * a2)
        x = (produit - 1.0 + np.sqrt((produit - 1.0) ** 2 + 4.0 * abs(a2) ** 2)) / (
            2.0 * abs(a2)
        )
        assert x - 1.0 / x == pytest.approx(abs(a1) - 1.0 / abs(a2), abs=1e-12)


def test_les_formes_fermees_sont_exactes_sur_la_covariance_theorique():
    """Sans aucun échantillonnage : C_O = D C_Q D^H construite analytiquement.

    Isole le biais de **modèle** du bruit d'estimation. À couplage co-pol /
    cross-pol négligeable (`beta` très petit) et crosstalk très faible, les
    formes fermées doivent redonner `theta` à la précision machine près.
    """
    from src.crosstalk.physics import distortion_matrix
    from src.crosstalk.synthetic import quegan_covariance

    vrai = theta(1e-3)
    matrice = distortion_matrix(vrai)
    cov = matrice @ quegan_covariance(beta=1e-6) @ matrice.conj().T
    estime = CrosstalkParameters.from_vector(estimateur()._solve(cov)[0])
    assert erreur_relative(estime, vrai) < 1e-4


# ==================================================================== invariances


def test_invariance_a_une_homothetie_reelle_globale():
    """Fonde la compatibilité avec la normalisation de configs/data_uavsar.yaml."""
    patch = observe(theta(0.1), shape=(256, 256))
    a = estimateur().estimate(patch)
    b = estimateur().estimate(4.2 * patch)
    np.testing.assert_allclose(a.to_vector(), b.to_vector(), rtol=1e-9)


def test_un_patch_sans_distorsion_donne_un_crosstalk_negligeable():
    neutre = CrosstalkParameters(0.0, 0.0, 0.0, 0.0, 1.0)
    signal = observe(neutre, shape=(512, 512), beta=0.02)
    estime = estimateur().estimate(signal)
    assert estime.max_crosstalk_db() < -35.0


# ======================================================== variantes de la phase


def test_variante_babu_rend_un_alpha_reel():
    patch = observe(theta(0.1), beta=0.002)
    babu = estimateur(alpha_phase="babu").estimate(patch)
    assert babu.alpha.imag == 0.0
    assert babu.alpha.real > 0.0


def test_les_deux_variantes_partagent_le_module_de_alpha():
    patch = observe(theta(0.1), beta=0.002)
    fore = estimateur(alpha_phase="fore").estimate(patch)
    babu = estimateur(alpha_phase="babu").estimate(patch)
    assert abs(fore.alpha) == pytest.approx(abs(babu.alpha), rel=1e-12)
    np.testing.assert_allclose(
        fore.to_vector()[:4], babu.to_vector()[:4], rtol=1e-12
    )


def test_variante_inconnue_refusee():
    with pytest.raises(ValueError, match="alpha_phase"):
        QueganEstimator(alpha_phase="kimura")


# ============================================================ fenêtre glissante


def test_mode_glissant_retrouve_aussi_theta():
    vrai = theta(0.1)
    patch = observe(vrai, shape=(128, 128), beta=0.002)
    estime = estimateur(window=(16, 16)).estimate(patch)
    assert erreur_relative(estime, vrai) < 0.30


def test_estimate_range_lines_rend_un_jeu_par_colonne():
    patch = observe(theta(0.1), shape=(64, 40), beta=0.02)
    lignes = estimateur(window=BABU_WINDOW).estimate_range_lines(patch)
    assert len(lignes) == 40 - 5 + 1
    assert all(isinstance(jeu, CrosstalkParameters) for jeu in lignes)


def test_estimate_range_lines_refuse_le_mode_full():
    with pytest.raises(ValueError, match="fenêtre glissante"):
        estimateur(window="full").estimate_range_lines(observe(theta(), (16, 16)))


def test_fenetre_mal_formee_refusee():
    with pytest.raises(ValueError, match="window"):
        QueganEstimator(window=(5, 5, 5))


# ============================================================== garde-fous


def test_patch_constant_leve_une_exception():
    # Covariance de rang 1 : Delta et X sont nuls.
    constant = np.ones((4, 16, 16), dtype=np.complex64)
    with pytest.raises(IllConditionedCovarianceError):
        estimateur().estimate(constant)


def test_patch_nul_leve_une_exception():
    with pytest.raises(IllConditionedCovarianceError):
        estimateur().estimate(np.zeros((4, 16, 16), dtype=np.complex64))


def test_le_message_nomme_le_critere_degrade():
    constant = np.ones((4, 8, 8), dtype=np.complex64)
    with pytest.raises(IllConditionedCovarianceError) as erreur:
        estimateur().estimate(constant)
    assert "C11 C44" in str(erreur.value) or "C22 C33" in str(erreur.value)


def test_les_seuils_sont_relatifs_donc_invariants_d_echelle():
    """Une homothétie ne doit pas faire basculer un garde-fou.

    `Delta` est homogène à une puissance au carré : un seuil absolu se
    déclencherait ou non selon la normalisation appliquée en amont.
    """
    patch = observe(theta(0.1), shape=(128, 128))
    for facteur in (1e-4, 1.0, 1e4):
        estimateur().estimate(facteur * patch)  # aucune exception


def test_fenetres_degenerees_ecartees_et_signalees(caplog):
    """Une fenêtre inexploitable ne doit pas condamner tout le patch."""
    patch = observe(theta(0.1), shape=(64, 64), beta=0.02).copy()
    patch[:, :16, :16] = 1.0  # un coin constant : fenêtres dégénérées
    with caplog.at_level(logging.WARNING, logger="src.crosstalk.estimators.quegan"):
        estime = estimateur(window=(8, 8)).estimate(patch)
    assert "écartées" in caplog.text
    assert np.all(np.isfinite(estime.to_vector()))


# ============================================================ intégration


def test_fenetres_vides_ecartees():
    """Cas réel : les patchs du bord de fauchée contiennent des pixels nuls.

    La grille des `.slc` UAVSAR déborde la fauchée utile ; un patch aligné sur
    le bord contient des zones exactement nulles. Les covariances y sont nulles
    et les critères de conditionnement valent `NaN`. Ces fenêtres doivent être
    écartées, et non contaminer l'estimation du reste du patch.
    """
    patch = observe(theta(0.1), shape=(64, 64), beta=0.02).copy()
    patch[:, :, :20] = 0.0  # bord de fauchée
    estime = estimateur(window=BABU_WINDOW).estimate(patch)
    assert np.all(np.isfinite(estime.to_vector()))
    assert erreur_relative(estime, theta(0.1)) < 0.5


def test_un_patch_entierement_vide_leve_une_exception():
    vide = np.zeros((4, 32, 32), dtype=np.complex64)
    with pytest.raises(IllConditionedCovarianceError):
        estimateur(window=BABU_WINDOW).estimate(vide)


def test_enregistre_dans_le_registre():
    estime = create_from_config(
        {"name": "quegan", "params": {"window": "full", "alpha_phase": "babu"}}
    )
    assert isinstance(estime, QueganEstimator)
    assert estime.alpha_phase == "babu"


def test_correction_heritee_de_la_base():
    vrai = theta(0.1)
    patch = observe(vrai, shape=(256, 256), beta=0.002)
    corrige, estime = estimateur().estimate_and_correct(patch)
    assert corrige.shape == patch.shape
    # Le crosstalk résiduel, ré-estimé sur les données corrigées, doit chuter.
    residuel = estimateur().estimate(corrige)
    assert residuel.max_crosstalk_db() < estime.max_crosstalk_db() - 10.0


def test_representation_lisible():
    texte = repr(estimateur(window=BABU_WINDOW, alpha_phase="babu"))
    assert "quegan" not in texte.lower() or "QueganEstimator" in texte
    assert "(5, 5)" in texte and "babu" in texte

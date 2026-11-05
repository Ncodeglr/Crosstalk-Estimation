"""Tests de l'estimateur de Quegan, dont le test de cohérence bout en bout."""

import numpy as np
import pytest

from src.crosstalk.baselines import create_from_config
from src.crosstalk.baselines.quegan import QueganEstimator
from src.crosstalk.polarimetry.parameters import CrosstalkParameters
from src.crosstalk.polarimetry.matrix import apply_distortion
from synthetic import ScatteringParameters, sample_scattering

#: Direction de crosstalk de référence, mise à l'échelle par `epsilon`.
DIRECTION = np.array([0.8 + 0.5j, -0.6 + 0.9j, 1.0 - 0.3j, -0.4 - 0.7j])
ALPHA = 1.03 * np.exp(1j * np.deg2rad(27.0))


def theta(epsilon: float = 0.1) -> CrosstalkParameters:
    """Jeu de référence à `max|u,v,w,z| ~ 0.108 * epsilon / 0.1`."""
    return CrosstalkParameters.from_vector([*(epsilon * DIRECTION), ALPHA])


def observe(params, shape=(1024, 1024), beta=0.05, graine=0, **kwargs):
    """Observation synthétique sans bruit, en ordre modèle : `s' = D s`."""
    signal = sample_scattering(ScatteringParameters(beta=beta, **kwargs), shape, rng=np.random.default_rng(graine)
    )
    return apply_distortion(signal, params)


def estimateur(**kwargs) -> QueganEstimator:
    return QueganEstimator(**kwargs)


def erreur_relative(estime: CrosstalkParameters, vrai: CrosstalkParameters) -> float:
    a, b = estime.to_vector()[:4], vrai.to_vector()[:4]
    return float((np.abs(a - b) / np.abs(b)).max())


# ============================================================ cohérence bout en bout


def test_bout_en_bout_couplage_faible():
    """Le test central : `s` -> `s' = D s` -> `theta` ré-estimé.

    Le couplage co-pol/cross-pol est réduit (`beta = 0.002`, soit -27 dB) pour
    que le biais structurel de la méthode, quantifié plus bas, reste sous le
    pour-cent et laisse voir la justesse des formules elles-mêmes.
    """
    vrai = theta(0.1)
    estime = estimateur().estimate(observe(vrai, beta=0.002))
    assert erreur_relative(estime, vrai) < 0.03


def test_alpha_est_retrouve_en_module_et_en_phase():
    """Le module suit Babu éq. 15, la phase Fore éq. 25 (`alpha1 / |alpha1|`).

    Le jeu de référence porte une phase de 27°. Sans le facteur de Fore,
    l'estimation rendrait 0°, comme l'équation 15 de Babu prise à la lettre.
    """
    estime = estimateur().estimate(observe(theta(0.1), beta=0.002))
    assert abs(estime.alpha) == pytest.approx(abs(ALPHA), rel=0.02)
    assert np.degrees(np.angle(estime.alpha)) == pytest.approx(27.0, abs=1.0)


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

    En développant les équations 15 à 18 de Babu et al. (2022), le module rendu est
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
    """Sans aucun échantillonnage : C_S' = D C_Q D^H construite analytiquement.

    Isole le biais de **modèle** du bruit d'estimation. À couplage co-pol /
    cross-pol négligeable (`beta` très petit) et crosstalk très faible, les
    formes fermées doivent redonner `theta` à la précision machine près.
    """
    from src.crosstalk.polarimetry.matrix import distortion_matrix

    vrai = theta(1e-3)
    matrice = distortion_matrix(vrai)
    cov = matrice @ ScatteringParameters(beta=1e-6).to_covariance() @ matrice.conj().T
    estime = CrosstalkParameters.from_vector(estimateur().solve(cov))
    assert erreur_relative(estime, vrai) < 1e-4


@pytest.mark.parametrize("beta", [0.05, 0.134])
def test_le_biais_est_le_terme_croise_neglige_par_les_formes_fermees(beta):
    """Le biais de Quegan a une origine identifiée, pas seulement un ordre de grandeur.

    Au premier ordre en crosstalk, avec `P_X = <|S_HV|^2>` la puissance croisée
    vraie ::

        C31 = u C11 + v C41 + P_X (e^{i arg alpha} v* + |alpha| w*) + O(theta^3)
        C34 = u C14 + v C44 + P_X (e^{i arg alpha} u* + |alpha| z*) + O(theta^3)
        C21 = z C11 + w C41 + P_X (v*/|alpha| + e^{-i arg alpha} w*) + O(theta^3)
        C24 = z C14 + w C44 + P_X (u*/|alpha| + e^{-i arg alpha} z*) + O(theta^3)

    Les formes fermées résolvent ces systèmes 2x2 sans les termes en `P_X` : le
    biais doit donc être la solution du même système appliquée à ces termes.
    Covariance théorique, sans échantillonnage ; crosstalk à -39 dB pour rendre
    le reste en `O(theta^3)` négligeable. Puissances co-polaires inégales et
    `alpha` loin de 1 en module comme en phase, pour que chaque facteur compte.
    `beta = 0.134` est de l'ordre de la médiane mesurée sur la pile (0,119).
    """
    from src.crosstalk.polarimetry.matrix import distortion_matrix

    vrai = CrosstalkParameters.from_vector(
        [*(1e-2 * DIRECTION), 1.25 * np.exp(-1j * np.deg2rad(23.0))]
    )
    u, v, w, z, alpha = vrai.to_vector()
    sigma_hh, sigma_vv = 2.0, 3.0
    p_x = beta * (sigma_hh + sigma_vv) / 2.0
    matrice = distortion_matrix(vrai)
    cov = (
        matrice
        @ ScatteringParameters(sigma_hh, sigma_vv, 0.5 * np.exp(0.3j) * np.sqrt(sigma_hh * sigma_vv), p_x).to_covariance()
        @ matrice.conj().T
    )
    biais = estimateur().solve(cov)[:4] - vrai.to_vector()[:4]

    phase, module = np.exp(1j * np.angle(alpha)), abs(alpha)
    systeme = np.array([[cov[0, 0], cov[3, 0]], [cov[0, 3], cov[3, 3]]])
    du, dv = np.linalg.solve(systeme, p_x * np.array([
        phase * np.conj(v) + module * np.conj(w),
        phase * np.conj(u) + module * np.conj(z),
    ]))
    dz, dw = np.linalg.solve(systeme, p_x * np.array([
        np.conj(v) / module + np.conj(w) / phase,
        np.conj(u) / module + np.conj(z) / phase,
    ]))
    predit = np.array([du, dv, dw, dz])

    assert np.abs(biais - predit).max() < 1e-2 * np.abs(biais).max()


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


# ============================================================ intégration


def test_un_patch_de_bord_de_fauchee_reste_estimable():
    """Cas réel : les patchs du bord de fauchée contiennent des pixels nuls.

    La grille des `.slc` UAVSAR déborde la fauchée utile ; un patch aligné sur
    le bord contient des zones exactement nulles. Elles ne font que diluer la
    covariance du patch d'un facteur réel, auquel les formes fermées sont
    insensibles : l'estimation est celle de la partie utile.
    """
    patch = observe(theta(0.1), shape=(64, 64), beta=0.02).copy()
    patch[:, :, :20] = 0.0  # bord de fauchée
    estime = estimateur().estimate(patch)
    np.testing.assert_allclose(
        estime.to_vector(), estimateur().estimate(patch[:, :, 20:]).to_vector(), rtol=1e-9
    )


def test_enregistre_dans_le_registre():
    assert isinstance(create_from_config({"name": "quegan"}), QueganEstimator)


def test_correction_heritee_de_la_base():
    vrai = theta(0.1)
    patch = observe(vrai, shape=(256, 256), beta=0.002)
    corrige, estime = estimateur().estimate_and_correct(patch)
    assert corrige.shape == patch.shape
    # Le crosstalk résiduel, ré-estimé sur les données corrigées, doit chuter.
    residuel = estimateur().estimate(corrige)
    assert residuel.max_crosstalk_db() < estime.max_crosstalk_db() - 10.0


def test_representation_lisible():
    texte = repr(estimateur())
    assert "QueganEstimator" in texte and "tol" in texte

"""
Tests du générateur de signal vrai, couche par couche :

1. les conventions partagées avec le reste du paquet ;
2. `ScatteringParameters` : conversion des types, validation, immuabilité, corrélation HH-VV ;
3. `to_covariance` : la matrice C_Q de Fore 2015, éq. 17 ;
4. `sample_scattering` : un patch dont la covariance attendue est C_Q.

Les tests statistiques utilisent des graines fixes et des tolérances fixées d'après l'erreur
d'estimation attendue, de l'ordre de sigma / sqrt(N) pour N pixels : ils ne peuvent pas échouer au hasard.
"""

import dataclasses

import numpy as np
import pytest

from src.crosstalk.polarimetry.covariance import polarimetric_covariance
from src.crosstalk.polarimetry.matrix import POL_ORDER, distortion_matrix
from src.crosstalk.polarimetry.parameters import CrosstalkParameters
from synthetic import (
    IDX_HH,
    IDX_HV,
    IDX_VH,
    IDX_VV,
    ScatteringParameters,
    sample_scattering,
)

#: Une scène non triviale : puissances différentes, corrélation HH-VV complexe de module 0,5.
SCENE = ScatteringParameters(
    sigma_hhhh=2.0, sigma_vvvv=3.0, sigma_hhvv=0.5 * np.sqrt(6.0) * np.exp(0.4j), beta=0.1
)


# ------------------------------------------------------------------ conventions


def test_les_indices_locaux_suivent_l_ordre_du_modele():
    """
    Le garde-fou de la double déclaration.

    `synthetic` redéclare ses indices au lieu de les importer de `matrix` ;
    ce test est ce qui empêche les deux déclarations de diverger en silence —
    une divergence qui ne lèverait aucune exception et se contenterait de
    renommer les paramètres estimés.
    """
    assert (IDX_HH, IDX_HV, IDX_VH, IDX_VV) == tuple(
        POL_ORDER.index(pol) for pol in ("HH", "HV", "VH", "VV")
    )


# --------------------------------------------------------- ScatteringParameters


def test_valeurs_par_defaut():
    p = ScatteringParameters()
    assert (p.sigma_hhhh, p.sigma_vvvv, p.sigma_hhvv, p.beta) == (1.0, 1.0, 0.6 + 0j, 0.05)


def test_les_types_sont_normalises():
    """Un scalaire numpy ou un entier devient un float Python, et sigma_hhvv un complex."""
    p = ScatteringParameters(
        sigma_hhhh=np.float32(2.0), sigma_vvvv=3, sigma_hhvv=np.complex64(0.5), beta=np.float64(0.1)
    )
    assert type(p.sigma_hhhh) is float and type(p.sigma_vvvv) is float and type(p.beta) is float
    assert type(p.sigma_hhvv) is complex


@pytest.mark.parametrize("champ", ["sigma_hhhh", "sigma_vvvv", "beta"])
def test_une_puissance_negative_est_refusee(champ):
    with pytest.raises(ValueError, match="positives"):
        ScatteringParameters(**{champ: -0.1})


def test_cauchy_schwarz_violee_est_refusee():
    """|sigma_hhvv|^2 > sigma_hhhh * sigma_vvvv : ce n'est pas une covariance."""
    with pytest.raises(ValueError, match="sigma_hhvv"):
        ScatteringParameters(sigma_hhhh=1.0, sigma_vvvv=1.0, sigma_hhvv=1.2)


def test_l_egalite_est_refusee_car_delta_s_annule():
    """
    |sigma_hhvv|^2 = sigma_hhhh * sigma_vvvv est une covariance valide (HH et VV parfaitement
    corrélés), mais elle annule Delta, le dénominateur de Quegan : refusée elle aussi.
    """
    with pytest.raises(ValueError, match="sigma_hhvv"):
        ScatteringParameters(sigma_hhhh=1.0, sigma_vvvv=1.0, sigma_hhvv=1.0)


def test_juste_sous_la_limite_le_jeu_est_accepte():
    p = ScatteringParameters(sigma_hhhh=1.0, sigma_vvvv=1.0, sigma_hhvv=1.0 - 1e-9)
    assert abs(p.correlation_hhvv) < 1.0


def test_le_jeu_est_immuable():
    p = ScatteringParameters()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.beta = 0.2


def test_replace_repasse_par_la_validation():
    """`dataclasses.replace` crée un nouvel objet : il est validé comme les autres."""
    p = ScatteringParameters()
    assert dataclasses.replace(p, beta=0.2).beta == 0.2
    with pytest.raises(ValueError, match="positives"):
        dataclasses.replace(p, beta=-0.2)


def test_la_correlation_hhvv_est_la_covariance_normalisee():
    attendu = SCENE.sigma_hhvv / np.sqrt(SCENE.sigma_hhhh * SCENE.sigma_vvvv)
    assert SCENE.correlation_hhvv == pytest.approx(attendu)
    assert abs(SCENE.correlation_hhvv) == pytest.approx(0.5)
    assert np.angle(SCENE.correlation_hhvv) == pytest.approx(0.4)


def correlation_observee(niveau_db: float) -> complex:
    """Corrélation HH-VV lue sur la covariance observée exacte D C_Q D^H, pour un crosstalk donné."""
    amplitude = 10 ** (niveau_db / 20)
    theta = CrosstalkParameters(
        *(amplitude * np.exp(1j * phase) for phase in (0.3, -1.1, 2.0, -2.5)), 1.18 * np.exp(0.4j)
    )
    d = distortion_matrix(theta)
    observee = d @ SCENE.to_covariance() @ d.conj().T
    return observee[IDX_HH, IDX_VV] / np.sqrt(observee[IDX_HH, IDX_HH].real * observee[IDX_VV, IDX_VV].real)


def test_la_correlation_n_est_modifiee_par_le_crosstalk_qu_au_second_ordre():
    """
    Quegan 1994, éq. 40 : le module se lit sur les données observées, |C14| / sqrt(C11 C44).
    L'invariance n'est exacte qu'au premier ordre : l'écart, module comme phase, est du second ordre.
    Passer de -20 à -30 dB divise l'amplitude du crosstalk par sqrt(10), donc l'écart par 10.
    """
    vraie = SCENE.correlation_hhvv
    ecart_module = {db: abs(abs(correlation_observee(db)) - abs(vraie)) for db in (-20.0, -30.0)}
    ecart_phase = {db: abs(np.angle(correlation_observee(db) / vraie)) for db in (-20.0, -30.0)}
    assert ecart_module[-20.0] / ecart_module[-30.0] == pytest.approx(10.0, rel=0.1)
    assert ecart_phase[-20.0] / ecart_phase[-30.0] == pytest.approx(10.0, rel=0.1)
    # Ordre de grandeur pratique : à -20 dB, quelques pour cent du module et quelques degrés au plus.
    assert ecart_module[-20.0] < 0.02 * abs(vraie)
    assert np.degrees(ecart_phase[-20.0]) < 3.0


# ----------------------------------------------------------------- to_covariance


def test_structure_de_la_covariance_de_quegan():
    cov = SCENE.to_covariance()
    # Blocs co-pol / cross-pol nuls : symétrie de réflexion (Quegan, hypothèse 3).
    for i in (IDX_HH, IDX_VV):
        for j in (IDX_VH, IDX_HV):
            assert cov[i, j] == 0.0
            assert cov[j, i] == 0.0
    assert cov[IDX_HH, IDX_HH] == SCENE.sigma_hhhh
    assert cov[IDX_VV, IDX_VV] == SCENE.sigma_vvvv
    assert cov[IDX_HH, IDX_VV] == SCENE.sigma_hhvv
    assert cov[IDX_VV, IDX_HH] == np.conj(SCENE.sigma_hhvv)
    # Bloc croisé constant : S_HV = S_VH (Quegan, hypothèse 2).
    for i in (IDX_HV, IDX_VH):
        for j in (IDX_HV, IDX_VH):
            assert cov[i, j] == SCENE.beta


def test_la_covariance_est_hermitienne_sdp_et_de_rang_3():
    cov = SCENE.to_covariance()
    np.testing.assert_allclose(cov, cov.conj().T)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-12)
    assert np.linalg.matrix_rank(cov) == 3  # les canaux croisés sont confondus


def test_l_ordre_de_fore_donne_la_meme_matrice():
    """
    Fore écrit l'éq. 17 dans l'ordre [HH, VH, HV, VV]. Le bloc croisé étant rempli de beta,
    échanger HV et VH laisse la matrice inchangée : elle est identique dans les deux ordres.
    """
    cov = SCENE.to_covariance()
    ordre_fore = [IDX_HH, IDX_VH, IDX_HV, IDX_VV]
    np.testing.assert_array_equal(cov[np.ix_(ordre_fore, ordre_fore)], cov)


# ------------------------------------------------------------- sample_scattering


def test_forme_et_dtype():
    rng = np.random.default_rng(0)
    assert sample_scattering(SCENE, (8, 9), rng).shape == (4, 8, 9)
    assert sample_scattering(SCENE, (8, 9), rng).dtype == np.complex128
    assert sample_scattering(SCENE, (8, 9), rng, dtype=np.complex64).dtype == np.complex64


def test_le_generateur_aleatoire_est_obligatoire():
    with pytest.raises(TypeError):
        sample_scattering(SCENE, (8, 8))


def test_reproductible_a_graine_egale():
    a = sample_scattering(SCENE, (8, 8), np.random.default_rng(1))
    b = sample_scattering(SCENE, (8, 8), np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_des_graines_differentes_donnent_des_patchs_differents():
    a = sample_scattering(SCENE, (8, 8), np.random.default_rng(1))
    b = sample_scattering(SCENE, (8, 8), np.random.default_rng(2))
    assert not np.allclose(a, b)


def test_reciprocite_stricte():
    """S_HV = S_VH au bit près, et non seulement en moyenne."""
    signal = sample_scattering(SCENE, (32, 32), np.random.default_rng(0))
    np.testing.assert_array_equal(signal[IDX_HV], signal[IDX_VH])


def test_symetrie_de_reflexion():
    """
    Corrélation co-pol / cross-pol nulle en espérance. Sur N = 512^2 pixels, l'écart-type d'un
    terme estimé vaut sqrt(sigma_co * beta / N), soit ~1e-3 ici : la tolérance de 5e-3 vaut ~5 écarts-types.
    """
    cov = polarimetric_covariance(sample_scattering(SCENE, (512, 512), np.random.default_rng(0)))
    for i in (IDX_HH, IDX_VV):
        for j in (IDX_HV, IDX_VH):
            assert abs(cov[i, j]) < 5e-3


def test_le_tirage_converge_vers_la_covariance_visee():
    """
    Sur N = 1024^2 pixels, l'écart-type d'un terme estimé est de l'ordre de sigma / sqrt(N) ~ 3e-3
    pour la plus forte puissance (3) : la tolérance de 1,5e-2 vaut ~5 écarts-types.
    """
    empirique = polarimetric_covariance(sample_scattering(SCENE, (1024, 1024), np.random.default_rng(0)))
    np.testing.assert_allclose(empirique, SCENE.to_covariance(), atol=1.5e-2)


def test_la_phase_de_la_correlation_copolaire_est_respectee():
    """
    Non-régression : la convention C_ij = <S_i S_j*> impose la conjugaison dans le tirage.

    Une corrélation co-polaire réelle ne distinguerait pas les deux conventions ;
    il faut une phase non nulle pour que l'erreur soit visible.
    """
    scene = ScatteringParameters(sigma_hhvv=0.6 * np.exp(1j * np.deg2rad(40.0)))
    mesure = polarimetric_covariance(sample_scattering(scene, (1024, 1024), np.random.default_rng(0)))
    assert np.angle(mesure[IDX_HH, IDX_VV], deg=True) == pytest.approx(40.0, abs=1.0)


def test_la_covariance_d_un_patch_fluctue_d_environ_un_sur_racine_de_n():
    """
    La covariance d'un patch fini n'est pas C_Q : elle fluctue autour, d'environ sigma / sqrt(N).
    Pour une puissance, l'estimateur est sans biais et son écart-type vaut exactement sigma / sqrt(N)
    (|z|^2 suit une loi exponentielle). Mesuré sur 400 patchs 64x64 : sigma_hhhh / 64.
    """
    rng = np.random.default_rng(0)
    puissances = [
        polarimetric_covariance(sample_scattering(SCENE, (64, 64), rng))[IDX_HH, IDX_HH].real
        for _ in range(400)
    ]
    assert np.mean(puissances) == pytest.approx(SCENE.sigma_hhhh, rel=0.01)
    assert np.std(puissances) == pytest.approx(SCENE.sigma_hhhh / 64, rel=0.15)


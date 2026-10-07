"""Tests du déséquilibre de canaux (Babu §3.2)."""

import numpy as np
import pytest

from src.crosstalk.imbalance import (
    ChannelImbalance,
    crosspol_imbalance,
    estimate_copol,
    estimate_crosspol,
)
from src.crosstalk.physics import POL_ORDER, apply_matrix
from src.crosstalk.synthetic import reflection_symmetric_scattering

HV, VH = POL_ORDER.index("HV"), POL_ORDER.index("VH")
HH, VV = POL_ORDER.index("HH"), POL_ORDER.index("VV")

VRAI = ChannelImbalance(f=1.15, g=0.93, phi_t=np.deg2rad(12.0), phi_r=np.deg2rad(-31.0))


def desequilibre(signal: np.ndarray, jeu: ChannelImbalance) -> np.ndarray:
    """Applique Babu, équation 7, dans le sens direct : `S'` vers `O`."""
    return apply_matrix(np.linalg.inv(jeu.correction_matrix()), signal, channel_axis=0)


# ------------------------------------------------------------- matrice


def test_matrice_diagonale_dans_l_ordre_des_polarisations():
    m = VRAI.correction_matrix()
    assert np.allclose(m, np.diag(np.diag(m)))
    attendu = {
        "HH": 1.0,
        "HV": 1.0 / (VRAI.f * VRAI.g * np.exp(1j * VRAI.phi_t)),
        "VH": 1.0 / (VRAI.f / VRAI.g * np.exp(1j * VRAI.phi_r)),
        "VV": 1.0 / (VRAI.f**2 * np.exp(1j * (VRAI.phi_t + VRAI.phi_r))),
    }
    for i, pol in enumerate(POL_ORDER):
        assert m[i, i] == pytest.approx(attendu[pol])


def test_jeu_neutre_donne_l_identite():
    assert np.allclose(ChannelImbalance(1.0, 1.0, 0.0, 0.0).correction_matrix(), np.eye(4))


@pytest.mark.parametrize("f, g", [(0.0, 1.0), (1.0, -0.5), (np.nan, 1.0), (1.0, np.inf)])
def test_desequilibre_non_positif_refuse(f, g):
    with pytest.raises(ValueError):
        ChannelImbalance(f, g, 0.0, 0.0)


def test_phase_non_finie_refusee():
    with pytest.raises(ValueError):
        ChannelImbalance(1.0, 1.0, np.nan, 0.0)


def test_from_estimates_separe_les_phases():
    jeu = ChannelImbalance.from_estimates(
        f=VRAI.f, phi_somme=VRAI.phi_t + VRAI.phi_r,
        g=VRAI.g, phi_difference=VRAI.phi_t - VRAI.phi_r,
    )
    assert jeu.phi_t == pytest.approx(VRAI.phi_t)
    assert jeu.phi_r == pytest.approx(VRAI.phi_r)


# ------------------------------------------------------------- estimation


def test_copol_retrouve_f_et_la_somme_des_phases():
    rng = np.random.default_rng(0)
    # Trièdres : S = sqrt(sigma) I, d'amplitude et de phase quelconques.
    a = rng.uniform(20, 80, 10) * np.exp(1j * rng.uniform(-np.pi, np.pi, 10))
    triedres = np.zeros((4, 10), dtype=np.complex128)
    triedres[HH] = triedres[VV] = a
    observe = desequilibre(triedres, VRAI)
    f, phi_somme = estimate_copol(observe[HH], observe[VV])
    assert f == pytest.approx(VRAI.f)
    assert phi_somme == pytest.approx(VRAI.phi_t + VRAI.phi_r)


def test_crosspol_retrouve_g_et_la_difference_des_phases():
    rng = np.random.default_rng(1)
    observe = desequilibre(reflection_symmetric_scattering((200_000,), rng=rng), VRAI)
    g, phi_difference = estimate_crosspol(observe[HV], observe[VH])
    # S_hv = S_vh exactement : l'estimation est exacte, au bruit numérique près.
    assert g == pytest.approx(VRAI.g, rel=1e-9)
    assert phi_difference == pytest.approx(VRAI.phi_t - VRAI.phi_r, abs=1e-9)


def test_aller_retour_complet_retablit_la_reciprocite():
    rng = np.random.default_rng(2)
    observe = desequilibre(reflection_symmetric_scattering((200_000,), rng=rng), VRAI)
    a = rng.uniform(20, 80, 10) * np.exp(1j * rng.uniform(-np.pi, np.pi, 10))
    triedres = np.zeros((4, 10), dtype=np.complex128)
    triedres[HH] = triedres[VV] = a
    pics = desequilibre(triedres, VRAI)

    jeu = ChannelImbalance.from_estimates(
        *estimate_copol(pics[HH], pics[VV]), *estimate_crosspol(observe[HV], observe[VH])
    )
    corrige = apply_matrix(jeu.correction_matrix(), observe, channel_axis=0)

    assert np.mean(np.abs(corrige[HV]) ** 2) == pytest.approx(np.mean(np.abs(corrige[VH]) ** 2))
    assert np.angle(np.mean(corrige[HV] * np.conj(corrige[VH]))) == pytest.approx(0.0, abs=1e-9)
    triedres_corriges = apply_matrix(jeu.correction_matrix(), pics, channel_axis=0)
    assert np.allclose(triedres_corriges[VV], triedres_corriges[HH])


def test_f_robuste_a_un_reflecteur_degrade():
    hh = np.ones(9, dtype=np.complex128)
    vv = np.full(9, 1.2**2, dtype=np.complex128)
    vv[0] = 50.0  # cible parasite
    f, _ = estimate_copol(hh, vv)
    assert f == pytest.approx(1.2)


def test_phase_copol_sans_repliement():
    # Des phases de part et d'autre de +-pi : une moyenne arithmétique rendrait ~0.
    phases = np.deg2rad([178.0, -178.0, 176.0, -176.0])
    f, phi = estimate_copol(np.ones(4), np.exp(1j * phases))
    assert abs(phi) == pytest.approx(np.pi)


def test_moments_et_tableaux_concordent():
    rng = np.random.default_rng(3)
    hv = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    vh = 0.8 * hv * np.exp(0.3j)
    assert estimate_crosspol(hv, vh) == pytest.approx(
        crosspol_imbalance(np.mean(abs(hv) ** 2), np.mean(abs(vh) ** 2), np.mean(hv * np.conj(vh)))
    )


@pytest.mark.parametrize(
    "appel",
    [
        lambda: estimate_copol(np.array([]), np.array([])),
        lambda: estimate_copol(np.zeros(3), np.ones(3)),
        lambda: estimate_copol(np.ones(3), np.ones(4)),
        lambda: estimate_crosspol(np.ones(3), np.ones(4)),
        lambda: crosspol_imbalance(0.0, 1.0, 0.5),
    ],
)
def test_entrees_degenerees_refusees(appel):
    with pytest.raises(ValueError):
        appel()

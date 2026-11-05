"""
Tests de la loi d'injection du banc d'essai :

1. CrosstalkSampler : valeurs par défaut, conversion des types, refus des lois invalides, immuabilité ;
2. CrosstalkSampler.generate : enveloppe et niveau commun des termes, proportion de patchs bruts, uniformité en dB du niveau et
   de |alpha|, uniformité des phases, inversibilité de D.

Les tests statistiques utilisent des graines fixes et des tolérances de plusieurs écarts-types : ils ne peuvent pas
échouer au hasard.
"""

import dataclasses

import numpy as np
import pytest

from src.crosstalk.injection.crosstalk_sampler import NEUTRE, CrosstalkSampler
from src.crosstalk.polarimetry.matrix import DEFAULT_TOL, apply_distortion, distortion_matrix
from src.crosstalk.polarimetry.parameters import CROSSTALK_NAMES


def tirages(sampler: CrosstalkSampler, n: int = 4000, graine: int = 0):
    rng = np.random.default_rng(graine)
    return [sampler.generate(rng) for _ in range(n)]


def injectes(sampler: CrosstalkSampler, n: int = 4000, graine: int = 0):
    """Les tirages qui ne sont pas le theta_k neutre."""
    return [theta for theta in tirages(sampler, n, graine) if theta != NEUTRE]


def niveaux_db(thetas) -> np.ndarray:
    """(n, 4) : le module de u, v, w, z en dB."""
    return np.array([[theta.crosstalk_db()[nom] for nom in CROSSTALK_NAMES] for theta in thetas])


def patch_de(graine: int = 0) -> np.ndarray:
    rng = np.random.default_rng(graine)
    return (rng.normal(size=(4, 16, 16)) + 1j * rng.normal(size=(4, 16, 16))).astype(np.complex64)


# ----------------------------------------------------------------- CrosstalkSampler


def test_valeurs_par_defaut():
    sampler = CrosstalkSampler()
    assert sampler.crosstalk_db == (-35.0, -10.0)
    assert sampler.term_spread_db == 4.0
    assert sampler.alpha_module_db == (-3.0, 3.0)
    assert sampler.alpha_phase_deg == (-30.0, 30.0)
    assert sampler.no_injection_rate == 0.0


def test_les_types_sont_normalises():
    sampler = CrosstalkSampler(crosstalk_db=[-30, -10], term_spread_db=2, alpha_module_db=(np.float32(-1), 1),
                       no_injection_rate=0)
    assert sampler.crosstalk_db == (-30.0, -10.0) and type(sampler.crosstalk_db) is tuple
    assert all(type(borne) is float for borne in sampler.alpha_module_db)
    assert type(sampler.term_spread_db) is float and type(sampler.no_injection_rate) is float


@pytest.mark.parametrize(
    "reglages",
    [
        dict(crosstalk_db=(-5.0, -40.0)),                         # bornes inversées
        dict(crosstalk_db=(-20.0, -20.0)),                        # intervalle vide
        dict(crosstalk_db=(-20.0, 0.0)),                          # 0 dB : D peut devenir singulière
        dict(crosstalk_db=(-np.inf, -5.0)),                       # borne infinie
        dict(alpha_module_db=(np.nan, 3.0)),                      # borne non définie
        dict(term_spread_db=-1.0),                                # écart négatif
        dict(crosstalk_db=(-20.0, -12.0), term_spread_db=4.0),    # plus de place pour le niveau commun
        dict(alpha_phase_deg=(-200.0, 30.0)),                     # phase hors de [-180, 180]
        dict(no_injection_rate=-0.1),                             # proportion négative
        dict(no_injection_rate=1.0),                              # plus aucun patch injecté
    ],
)
def test_une_loi_invalide_est_refusee(reglages):
    with pytest.raises(ValueError):
        CrosstalkSampler(**reglages)


@pytest.mark.parametrize(
    "reglages",
    [
        dict(crosstalk_db=-20.0),                     # un scalaire au lieu de deux bornes
        dict(crosstalk_db=(-40.0, -20.0, -5.0)),      # une borne de trop
        dict(alpha_module_db=(3.0,)),                 # une borne de moins
        dict(alpha_phase_deg="(-30, 30)"),            # une chaîne au lieu de deux nombres
        dict(alpha_phase_deg=((-30.0, 30.0),)),       # bornes imbriquées
    ],
)
def test_une_paire_de_bornes_mal_formee_est_refusee_avec_un_message_clair(reglages):
    """Une erreur de configuration doit nommer le champ fautif, pas laisser remonter une erreur Python générique."""
    with pytest.raises(ValueError, match="deux bornes"):
        CrosstalkSampler(**reglages)


def test_la_loi_est_immuable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CrosstalkSampler().crosstalk_db = (-30.0, -20.0)


# ----------------------------------------------------------------- CrosstalkSampler.generate


def test_les_tirages_restent_dans_les_bornes():
    sampler = CrosstalkSampler(crosstalk_db=(-30.0, -15.0), term_spread_db=3.0, alpha_module_db=(-1.0, 2.0),
                       alpha_phase_deg=(10.0, 40.0), no_injection_rate=0.0)
    for theta in tirages(sampler, n=1000):
        assert all(-30.0 <= db <= -15.0 for db in theta.crosstalk_db().values())
        assert -1.0 <= theta.alpha_db() <= 2.0
        assert 10.0 <= theta.alpha_phase_deg() <= 40.0


def test_les_quatre_termes_restent_autour_d_un_niveau_commun():
    """Deux termes d'un même tirage sont chacun à +-4 dB du niveau commun : ils diffèrent d'au plus 8 dB."""
    niveaux = niveaux_db(injectes(CrosstalkSampler()))
    assert np.all(niveaux.max(axis=1) - niveaux.min(axis=1) <= 2 * 4.0 + 1e-9)
    assert np.all((niveaux >= -35.0 - 1e-9) & (niveaux <= -10.0 + 1e-9))


def test_les_niveaux_des_termes_sont_correles_par_le_niveau_commun():
    """
    Variance du niveau commun (uniforme sur 17 dB) : 17^2 / 12 = 24,08 ; variance de l'écart propre à un terme
    (uniforme sur +-4 dB) : 8^2 / 12 = 5,33. Corrélation attendue entre deux termes : 24,08 / 29,42 = 0,82.
    """
    correlations = np.corrcoef(niveaux_db(injectes(CrosstalkSampler())).T)[np.triu_indices(4, k=1)]
    np.testing.assert_allclose(correlations, 0.82, atol=0.02)


def test_un_ecart_nul_donne_quatre_termes_de_meme_amplitude():
    """term_spread_db = 0 : la loi de Shi et al. (2020), des amplitudes identiques et des phases indépendantes."""
    for theta in tirages(CrosstalkSampler(term_spread_db=0.0, no_injection_rate=0.0), n=200):
        modules = np.abs(theta.to_vector()[:4])
        np.testing.assert_allclose(modules, modules[0], rtol=1e-12)


def test_le_niveau_commun_est_uniforme_en_db():
    """
    La moyenne en dB des quatre termes suit le niveau commun, uniforme sur [-31, -14] : quartiles -26,75, -22,5 et
    -18,25 dB. L'écart propre aux termes, moyenné sur quatre, ne les déplace que de quelques centièmes de dB.
    """
    moyennes = niveaux_db(injectes(CrosstalkSampler())).mean(axis=1)
    np.testing.assert_allclose(np.quantile(moyennes, [0.25, 0.5, 0.75]), [-26.75, -22.5, -18.25], atol=0.8)


def test_la_proportion_de_patchs_bruts_est_respectee():
    """p = 0,2 sur 5000 tirages : écart-type de la proportion 0,006 ; tolérance 0,02, soit plus de 3 écarts-types."""
    neutres = sum(theta == NEUTRE for theta in tirages(CrosstalkSampler(no_injection_rate=0.2), n=5000))
    assert neutres / 5000 == pytest.approx(0.2, abs=0.02)


def test_sans_patchs_bruts_aucun_tirage_n_est_neutre():
    assert NEUTRE not in tirages(CrosstalkSampler(no_injection_rate=0.0), n=1000)


def test_le_theta_neutre_laisse_le_patch_intact():
    patch = patch_de()
    np.testing.assert_allclose(apply_distortion(patch, NEUTRE), patch, rtol=1e-6)


def test_les_phases_des_termes_sont_uniformes():
    """Phase uniforme sur le cercle : la moyenne des exp(i phase) est nulle (environ 1/sqrt(n) sur n tirages)."""
    thetas = injectes(CrosstalkSampler())
    for nom in CROSSTALK_NAMES:
        assert abs(np.mean([np.exp(1j * np.angle(getattr(theta, nom))) for theta in thetas])) < 0.05


def test_le_module_de_alpha_est_uniforme_en_db():
    """Uniforme sur [-3, 3] dB : moyenne 0 dB, écart-type 6 / sqrt(12) = 1,73 dB."""
    db = np.array([theta.alpha_db() for theta in injectes(CrosstalkSampler())])
    assert db.min() >= -3.0 and db.max() <= 3.0
    assert db.mean() == pytest.approx(0.0, abs=0.1)
    assert db.std() == pytest.approx(6.0 / np.sqrt(12), rel=0.05)


def test_la_phase_de_alpha_est_uniforme_dans_ses_bornes():
    """Uniforme sur [-30, 30]° : moyenne 0°, écart-type 60 / sqrt(12) = 17,3°."""
    phases = np.array([theta.alpha_phase_deg() for theta in injectes(CrosstalkSampler())])
    assert phases.min() >= -30.0 and phases.max() <= 30.0
    assert phases.mean() == pytest.approx(0.0, abs=1.0)
    assert phases.std() == pytest.approx(60.0 / np.sqrt(12), rel=0.05)


def test_d_reste_inversible_a_l_extremite_de_la_loi():
    """
    Pour des modules <= m, |(uw-1)(vz-1)| >= (1 - m^2)^2 : 0,47 à -5 dB. La borne est approchée sans jamais être
    franchie, et reste très au-dessus du seuil de singularité.
    """
    m = 10 ** (-5.0 / 20)
    borne = (1 - m ** 2) ** 2
    assert borne > 1e6 * DEFAULT_TOL
    for theta in injectes(CrosstalkSampler(crosstalk_db=(-14.0, -5.0)), n=2000):
        u, v, w, z, _ = theta.to_vector()
        assert abs((u * w - 1) * (v * z - 1)) >= borne - 1e-12
        assert np.all(np.isfinite(np.linalg.inv(distortion_matrix(theta))))


def test_meme_graine_memes_tirages():
    assert tirages(CrosstalkSampler(), n=20, graine=3) == tirages(CrosstalkSampler(), n=20, graine=3)

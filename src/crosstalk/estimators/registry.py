"""Registre des méthodes d'estimation, pour instancier depuis une configuration.

Même forme que `cvnn.metrics_registry` du framework — un dictionnaire module,
un décorateur d'enregistrement, une fabrique par nom — afin que les baselines
se déclarent en YAML comme le reste du projet ::

    estimators:
      - {name: quegan, params: {window: full}}

Ajouter une méthode ne demande de toucher à aucun fichier existant : il suffit
de définir une classe décorée dans `crosstalk.estimators` ::

    @register_estimator
    class KimuraEstimator(CrosstalkEstimator):
        name = "kimura"
        def estimate(self, patch): ...

L'enregistrement a lieu à l'import du module qui définit la classe ; c'est
`crosstalk.estimators.__init__` qui importe les méthodes livrées.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

from .base import CrosstalkEstimator

_ESTIMATORS: dict[str, type[CrosstalkEstimator]] = {}

T = TypeVar("T", bound=type[CrosstalkEstimator])


def register_estimator(cls: T) -> T:
    """Décorateur : enregistre une classe sous son attribut `name`.

    Args:
        cls: sous-classe concrète de `CrosstalkEstimator`.

    Returns:
        La classe, inchangée.

    Raises:
        TypeError: si `cls` n'est pas une sous-classe de `CrosstalkEstimator`
            ou ne définit pas `name`.
        ValueError: si le nom est déjà pris par une autre classe.
    """
    if not (isinstance(cls, type) and issubclass(cls, CrosstalkEstimator)):
        raise TypeError(
            f"{cls!r} n'est pas une sous-classe de CrosstalkEstimator"
        )
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"{cls.__name__} doit définir un attribut de classe `name` non vide"
        )
    existant = _ESTIMATORS.get(name)
    if existant is not None and existant is not cls:
        raise ValueError(
            f"le nom {name!r} est déjà enregistré par {existant.__name__}"
        )
    _ESTIMATORS[name] = cls
    return cls


def available_estimators() -> list[str]:
    """Noms enregistrés, triés."""
    return sorted(_ESTIMATORS)


def get_estimator_class(name: str) -> type[CrosstalkEstimator]:
    """Classe enregistrée sous ce nom.

    Raises:
        KeyError: si le nom est inconnu ; le message liste les noms valides.
    """
    try:
        return _ESTIMATORS[name]
    except KeyError:
        raise KeyError(
            f"estimateur inconnu : {name!r}. Disponibles : "
            f"{available_estimators() or '(aucun)'}"
        ) from None


def create_estimator(name: str, **params: Any) -> CrosstalkEstimator:
    """Instancie un estimateur par son nom.

    Args:
        name: nom enregistré.
        **params: arguments passés au constructeur de la méthode.

    Raises:
        KeyError: si le nom est inconnu.
    """
    return get_estimator_class(name)(**params)


def create_from_config(config: Mapping[str, Any]) -> CrosstalkEstimator:
    """Instancie depuis un fragment de configuration.

    Args:
        config: mapping `{"name": ..., "params": {...}}`, `params` facultatif.

    Raises:
        KeyError: si `name` manque ou est inconnu.
        TypeError: si `params` n'est pas un mapping.
    """
    if "name" not in config:
        raise KeyError(f"configuration sans clé 'name' : {dict(config)!r}")
    params = config.get("params") or {}
    if not isinstance(params, Mapping):
        raise TypeError(
            f"'params' doit être un mapping, reçu {type(params).__name__}"
        )
    return create_estimator(config["name"], **params)

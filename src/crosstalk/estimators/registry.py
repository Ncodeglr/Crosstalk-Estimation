from __future__ import annotations
from typing import Any, Mapping, TypeVar
from .estimator import CrosstalkEstimator

_ESTIMATORS: dict[str, type[CrosstalkEstimator]] = {}
T = TypeVar("T", bound=type[CrosstalkEstimator])


def register_estimator(cls: T) -> T:
    """
    Décorateur : enregistre une classe sous son attribut name.
    Args: cls: sous-classe concrète de CrosstalkEstimator.
    Returns: La classe, inchangée.

    Raises:
        TypeError: si `cls` n'est pas une sous-classe de `CrosstalkEstimator`ou ne définit pas `name`.
        ValueError: si le nom est déjà pris par une autre classe.
    """
    if not (isinstance(cls, type) and issubclass(cls, CrosstalkEstimator)):
        raise TypeError(f"{cls!r} n'est pas une sous-classe de CrosstalkEstimator")
    
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} doit définir un attribut de classe `name` non vide")
    
    existant = _ESTIMATORS.get(name)
    if existant is not None and existant is not cls:
        raise ValueError(f"le nom {name!r} est déjà enregistré par {existant.__name__}")
    _ESTIMATORS[name] = cls
    return cls


def available_estimators() -> list[str]:
    """Noms enregistrés, triés."""
    return sorted(_ESTIMATORS)


def create_from_config(config: Mapping[str, Any]) -> CrosstalkEstimator:
    """
    À partir du nom, on crée un estimateur avec ses paramètres.

    Args: config: une entrée du YAML, `{name: ..., params: {...}}`. `params` est facultatif : absent ou vide (`None`), le constructeur reçoit ses valeurs par défaut.
    Raises:
        KeyError: si name manque, ou ne désigne aucune méthode enregistrée, le message liste alors les noms disponibles.
        TypeError: si params n'est pas un mapping.
    """
    if "name" not in config:
        raise KeyError(f"configuration sans clé 'name' : {dict(config)!r}")
    params = config.get("params") or {}
    if not isinstance(params, Mapping):
        raise TypeError(f"'params' doit être un mapping, reçu {type(params).__name__}")
    try:
        classe = _ESTIMATORS[config["name"]]
    except KeyError:
        raise KeyError(
            f"estimateur inconnu : {config['name']!r}. Disponibles : "
            f"{available_estimators() or '(aucun)'}"
        ) from None
    return classe(**params) #QueganEstimator(name='quegan', tol=1e-08)

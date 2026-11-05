"""
Un générateur aléatoire par patch, pour le banc d'essai par réinjection : patch_rng.

Rôle
----
Chaque patch doit recevoir le même theta_k à chaque époque et dans chaque processus du DataLoader, et le banc
comparatif doit pouvoir refaire exactement le même tirage pour évaluer Quegan sur les mêmes patchs injectés que le
réseau. patch_rng rend pour cela un générateur dont la graine combine seed et le contenu du patch :

    theta_k = CrosstalkSampler(**loi).generate(patch_rng(patch, seed))

Le module ne dépend que de numpy et de hashlib : le côté statistique peut l'importer sans torch ni cvnn.
"""

from __future__ import annotations
import hashlib
import numpy as np


def patch_rng(patch: np.ndarray, seed: int) -> np.random.Generator:
    """
    Générateur aléatoire propre à un patch : sa graine combine seed et une empreinte du contenu du patch.

    Le même patch donne toujours le même générateur, donc le même theta_k, quel que soit le processus ou l'époque.
    L'empreinte porte sur les octets, la forme et le type du tableau : un même patch en complex64 et en complex128
    donne deux générateurs différents. Le patch doit donc toujours être passé tel qu'il est lu.

    Args:
        patch: le patch, tel qu'il est lu.
        seed: entier positif ou nul. Le changer change toutes les injections : c'est une autre expérience.
    """
    if not isinstance(patch, np.ndarray):
        raise TypeError(f"patch de type {type(patch).__name__} : np.ndarray attendu")
    # Pour un tableau d'objets Python, tobytes() écrit des adresses mémoire et non des valeurs : l'empreinte
    # changerait d'un processus à l'autre. Seuls les types numériques donnent une empreinte déterministe.
    if patch.dtype.kind not in "biufc":
        raise TypeError(f"patch de type d'éléments {patch.dtype} : un type numérique est attendu (complex64 pour les .slc)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed de type {type(seed).__name__} : entier attendu")
    if seed < 0:
        raise ValueError(f"seed = {seed} : il doit être positif ou nul")
    
    #hashlib.blake2b est une fonction de hachage cryptographique qui produit un haché de 64 octets par défaut. 
    empreinte = hashlib.blake2b(digest_size=8) #Empreinte du patch, pour que le même patch donne toujours le même générateur, donc le même theta_k, quel que soit le processus ou l'époch.
    empreinte.update(f"{patch.dtype.str}{patch.shape}".encode()) #patch.dtype.str = '<c8' pour complex64 ou '<c16' pour complex128,  patch.shape = (4, H, W). On encode en bytes pour que hashlib puisse traiter la chaîne.
    empreinte.update(patch.tobytes(order='C')) #tobytes() écrit les octets dans l'ordre des index (ordre C), quelle que soit la disposition du tableau en mémoire : 8 octets par complexe en complex64, soit 8 x (Nb_channels x H_patch x W_patch) octets. On met à jour l'empreinte avec le contenu du patch.
    
    #empreinte.digest() rend les 8 octets de l'empreinte, qu'on convertit en entier avec int.from_bytes(..., "little") pour que le même patch donne toujours le même générateur. 
    #On combine avec seed pour que la graine globale change toutes les injections.
    #La méthode default_rng rend un générateur aléatoire indépendant, initialisé avec la graine combinée. 
    #Si on pose g = np.random.default_rng([13, 5601977483896678385]) alors g.random() rend toujours le même nombre aléatoire, quel que soit le processus ou l'époch.
    return np.random.default_rng([int(seed), int.from_bytes(empreinte.digest(), "little")]) #exemple : [13, 5601977483896678385] 
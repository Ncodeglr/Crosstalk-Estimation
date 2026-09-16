import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchcvnn.datasets.miccai2023 import MICCAI2023
from torchcvnn.datasets.miccai2023 import CINEView, AccFactor

# Importez votre classe existante (supposons qu'elle est dans miccai_dataset.py)
# from miccai_dataset import MICCAI2023, CINEView, AccFactor, kspace_to_image

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def generate_flat_dataset(
    root_dir: str,
    output_dir: str,
    target_count: int = 50000,
    img_size: int = None  # None = garder la taille originale
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- CORRECTION ICI ---
    # On utilise AccFactor.ACC8 car c'est le seul dossier présent sur votre disque.
    # Cela permet d'initialiser la classe sans crash.
    dataset = MICCAI2023(
        rootdir=root_dir, 
        view=CINEView.SAX, 
        acc_factor=AccFactor.ACC8 
    )
    
    count = 0
    print(f"Dataset chargé : {len(dataset)} patients trouvés.")
    print(f"Début de la génération dans {output_dir}...")
    
    for i in tqdm(range(len(dataset))):
        if count >= target_count:
            break
            
        try:
            # On récupère index 2 : fullsampled_data
            # Shape: (kx, ky, sc, sz, t) 
            # Note : Votre classe renvoie déjà transposé (kx, ky, sc, sz, t) ?
            # Vérifions le __getitem__ de votre classe :
            # fullsampled_data = load... .transpose(3, 4, 2, 1, 0) -> (kx, ky, sc, sz, t)
            # C'est bien ça.
            _, _, kspace_full = dataset[i]
        except Exception as e:
            # Parfois des fichiers .mat sont corrompus ou manquants
            continue

        # Manipulation des axes pour l'iFFT
        # On veut (sc, sz, t, kx, ky) pour appliquer l'iFFT sur les 2 derniers axes
        # Actuellement : (0:kx, 1:ky, 2:sc, 3:sz, 4:t)
        kspace_permuted = np.moveaxis(kspace_full, [0, 1], [-2, -1]) 
        
        # iFFT Complexe (passage du domaine fréquentiel au spatial)
        # fftshift est nécessaire avant et après l'ifft pour centrer le zéro
        image_complex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_permuted, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # Dimensions : (Coils, Slices, Time, Height, Width)
        n_coils, n_slices, n_times, h, w = image_complex.shape
        
        # Boucles d'extraction
        # On peut sub-sampler temporellement pour avoir plus de variété
        # ou tout prendre pour grossir le dataset rapidement.
        for coil in range(n_coils):
            if count >= target_count: break
            
            for sl in range(n_slices):
                if count >= target_count: break

                # On prend 1 frame temporelle sur 2 ou 3 pour éviter d'avoir 15 images quasi identiques
                for t in range(0, n_times, 2): 
                    if count >= target_count:
                        break
                    
                    img_raw = image_complex[coil, sl, t] # (H, W) complexe
                    
                    # Safety check dimensions - seulement si img_size est spécifié
                    if img_size is not None and (h < img_size or w < img_size):
                        continue
                        
                    # Crop & Normalisation
                    if img_size is not None:
                        img_cropped = crop_center(img_raw, img_size, img_size)
                    else:
                        img_cropped = img_raw  # Garder la taille originale
                    
                    # Normalisation robuste par le max du module
                    mag_max = np.abs(img_cropped).max() + 1e-9
                    img_normalized = img_cropped / mag_max
                    
                    # Conversion Tensor PyTorch (1, H, W)
                    # On garde le type complexe (complex64)
                    tensor = torch.from_numpy(img_normalized).unsqueeze(0).cfloat()
                    
                    save_name = output_path / f"mri_sax_{count:06d}.pt"
                    torch.save(tensor, save_name)
                    
                    count += 1

    print(f"Terminé ! {count} images générées.")

if __name__ == "__main__":
    # Chemin exact vu dans votre logs
    ROOT = "/home/qgabot/disque_deux/ChallegeData" 
    DEST = "../datasets/MICCAIChallenge2023/FlatDataset"
    
    # img_size=None pour garder la taille originale
    # img_size=128 pour cropper à 128x128
    generate_flat_dataset(ROOT, DEST, target_count=100000, img_size=None)
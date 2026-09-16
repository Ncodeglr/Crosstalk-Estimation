import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- YOUR LOCAL IMPORTS HERE ---
from cvnn.models.blocks import VariationalBottleneck
from cvnn.losses import ComplexELBOLoss
from cvnn.train import KLAnnealing

def generate_synthetic_complex_data(delta: float, angle_rad: float = math.pi / 4, num_samples: int = 5000):
    var_real = (1.0 + delta) / 2.0
    var_imag = (1.0 - delta) / 2.0
    
    # 1. Génération de l'ellipse alignée sur l'axe des réels
    real_part = torch.randn(num_samples) * math.sqrt(var_real)
    imag_part = torch.randn(num_samples) * math.sqrt(var_imag)
    data_complex_aligned = torch.complex(real_part, imag_part)
    
    # 2. Rotation dans le plan complexe (Introduction de la phase)
    rotation_factor = torch.exp(torch.tensor(1j * angle_rad))
    data_complex_rotated = data_complex_aligned * rotation_factor
    
    return data_complex_rotated.view(-1, 1, 1, 1)

def estimate_delta(z_array, angle_rad: float = math.pi / 4):
    """
    Estime le delta d'origine en compensant l'angle de rotation.
    P_rot = E[Z^2] = delta * e^(i * 2 * angle)
    On fait la rotation inverse pour retrouver la partie réelle.
    """
    z_c = z_array - np.mean(z_array)
    P = np.mean(z_c**2)
    C = np.mean(np.abs(z_c)**2)
    
    # Rotation inverse de la pseudo-covariance (-2 * angle)
    P_aligned = P * np.exp(-1j * 2 * angle_rad)
    return np.real(P_aligned) / C

def train_and_extract_latents(delta: float, model_type: str, epochs: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = generate_synthetic_complex_data(delta).to(device)
    
    is_standard = (model_type == "CL-CVAE")
    
    bottleneck = VariationalBottleneck(
        in_channels=1, latent_dim=1, input_size=1, activation="CReLU",
        normalization="None", layer_mode="complex",
        cov_mode="diag", standard_reparam=is_standard 
    ).to(device)
        
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=0.01)
    
    # KL Loss avec un vrai Annealer progressif
    loss_fn = ComplexELBOLoss(schedule="beta", beta_max=0.1, cov_mode="diag", standard_reparam=is_standard)
    annealer = KLAnnealing(kind="linear", warmup_steps=100) # Warmup crucial !
            
    bottleneck.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Passage Encodeur -> Décodeur
        x_hat, z, mu, p1, p2 = bottleneck(x_train)
                
        # Fake logsigmas pour la loss
        log_sigma2_dec = torch.tensor([-2.0], device=device, requires_grad=True) 
        
        outputs = (x_hat, mu, p1, p2, log_sigma2_dec, None)
        loss_val, _ = loss_fn(outputs, x_train, anneal=annealer)
        
        loss_val.real.backward()
        optimizer.step()
        
        # AVANCÉE DE L'ANNEALER (Il manquait cette ligne !)
        annealer.step()
        
    # 4. Évaluation
    bottleneck.eval()
    with torch.no_grad():
        x_hat_eval, z_eval, mu_eval, _, _ = bottleneck(x_train)
        
    return z_eval.squeeze().cpu().numpy(), x_hat_eval.squeeze().cpu().numpy()

# --- RUNNING THE EXPERIMENT ---
# On explore les deltas négatifs et positifs
# 1. Ajustement global des polices pour correspondre à LaTeX (Serif)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

deltas = [-0.9, -0.4, 0.0, 0.4, 0.9]
models = ["CL-CVAE", "WL-CVAE"]

# Figure en horizontal: 3 deltas par ligne, 5 lignes
fig, axes = plt.subplots(3, len(deltas), figsize=(3.2 * len(deltas), 12))

# Fonction pour colorer selon la proximité au delta vrai
def get_delta_color(delta_estimated, delta_true):
    error = abs(delta_estimated - delta_true)
    if error < 0.1:
        return '#2ecc71'  # Vert - excellent
    elif error < 0.2:
        return '#f39c12'  # Orange - bon
    else:
        return '#e74c3c'  # Rouge - mauvais

props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='darkgray', linewidth=1.5)
row_labels = ["Ground Truth", "CL-CVAE\n(Strict Linear)", "WL-CVAE\n(Widely Linear)"]

for i, delta in enumerate(deltas):
    print(f"Training models for Delta = {delta}...")
    
    x_true_tensor = generate_synthetic_complex_data(delta)
    x_true = x_true_tensor.squeeze().numpy()
    
    _, x_hat_cl = train_and_extract_latents(delta, "CL-CVAE")
    _, x_hat_wl = train_and_extract_latents(delta, "WL-CVAE")
    
    delta_true = estimate_delta(x_true)
    delta_cl = estimate_delta(x_hat_cl)
    delta_wl = estimate_delta(x_hat_wl)
    
    # --- Colonne 1 : Ground Truth ---
    axes[0, i].scatter(x_true.real, x_true.imag, alpha=0.3, s=5, c='gray')
    
    # Boîte avec delta estimé - plus grande et plus colorée
    color_true = get_delta_color(delta_true, delta)
    axes[0, i].text(0.5, 0.95, rf"$\hat{{\delta}} = {delta:.3f}$", 
                    transform=axes[0, i].transAxes, fontsize=14, fontweight='bold',
                    ha='center', verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor=color_true, alpha=0.85, edgecolor='black', linewidth=2))
        
    axes[0, i].axis('equal')
    axes[0, i].grid(True, linestyle='--', alpha=0.4)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    
    # --- Colonne 2 : CL-CVAE ---
    axes[1, i].scatter(x_hat_cl.real, x_hat_cl.imag, alpha=0.3, s=5, c='#3498db')
    
    color_cl = get_delta_color(delta_cl, delta)
    axes[1, i].text(0.5, 0.95, rf"$\hat{{\delta}} = {delta_cl:.3f}$", 
                    transform=axes[1, i].transAxes, fontsize=14, fontweight='bold',
                    ha='center', verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor=color_cl, alpha=0.85, edgecolor='black', linewidth=2))
    
    axes[1, i].axis('equal')
    axes[1, i].grid(True, linestyle='--', alpha=0.4)
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    
    # --- Colonne 3 : WL-CVAE ---
    axes[2, i].scatter(x_hat_wl.real, x_hat_wl.imag, alpha=0.3, s=5, c='#e74c3c')
    
    color_wl = get_delta_color(delta_wl, delta)
    axes[2, i].text(0.5, 0.95, rf"$\hat{{\delta}} = {delta_wl:.3f}$", 
                    transform=axes[2, i].transAxes, fontsize=14, fontweight='bold',
                    ha='center', verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor=color_wl, alpha=0.85, edgecolor='black', linewidth=2))
    
    axes[2, i].axis('equal')
    axes[2, i].grid(True, linestyle='--', alpha=0.4)
    axes[2, i].set_xticks([])
    axes[2, i].set_yticks([])


# bbox_inches='tight' est crucial pour recadrer parfaitement le PDF sans espaces blancs inutiles
plt.tight_layout(rect=[0.05, 0, 1, 0.96])
plt.savefig("synthetic_circularity_figure.pdf", bbox_inches='tight')
plt.close()
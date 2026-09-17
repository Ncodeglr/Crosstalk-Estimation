import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import math
import wandb
import torchcvnn.nn.modules as c_nn

# ==============================================================================
from cvnn.models.feature_extractors import MnistClassifier, MstarClassifier
from cvnn.data import get_dataloaders

def compute_dataset_impropriety(dataloader: torch.utils.data.DataLoader, device: str = 'cuda') -> float:
    """
    Compute the global Degree of Impropriety (Quotient de Circularité) of a dataset.
    Returns rho (ρ), a value between 0 (Perfectly Circular) and 1 (Totally Improper).
    """
    sum_z = 0.0 + 0j
    sum_z2 = 0.0 + 0j
    sum_abs_z2 = 0.0
    total_pixels = 0

    for batch in dataloader:
        # 1. Data extraction (handles different batch formats)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
            
        x = x.to(device)
        
        if not x.is_complex():
            raise ValueError("The tensor is not complex. If you are using a Dual-Real format, convert it with torch.complex(x[:,0], x[:,1]) first.")

        # 2. Accumulation of sums (using .item() to offload GPU memory)
        # x.sum()              -> For E[Z]
        # (x**2).sum()         -> For E[Z^2] (Pseudo-Variance)
        # (x.abs()**2).sum()   -> For E[|Z|^2] (Standard Variance)
        
        sum_z += x.sum().item()
        sum_z2 += (x ** 2).sum().item()
        sum_abs_z2 += (x.abs() ** 2).sum().item()
        total_pixels += x.numel()

    # 3. Calculation of Expectations
    if total_pixels == 0:
        return 0.0
    E_z = sum_z / total_pixels
    E_z2 = sum_z2 / total_pixels
    E_abs_z2 = sum_abs_z2 / total_pixels

    # 4. Calculation of Central Covariances
    # C = E[|Z - E[Z]|^2] = E[|Z|^2] - |E[Z]|^2
    # P = E[(Z - E[Z])^2] = E[Z^2] - (E[Z])^2
    C = E_abs_z2 - abs(E_z)**2
    P = E_z2 - E_z**2
    
    # 5. Circularity Quotient (Degree of Impropriety)
    # ρ = |P| / C, with ρ in [0, 1]
    rho = abs(P) / C
    
    print(f"--- Dataset Statistics ---")
    print(f"Variance (C)        : {C:.4f}")
    print(f"Pseudo-Variance (|P|): {abs(P):.4f}")
    print(f"Impropriety (ρ)     : {rho:.4f}")
    
    return rho

def compute_dataset_circular_linear_correlation(dataloader: torch.utils.data.DataLoader, device: str = 'cuda') -> float:
    """
    Calculates the global Circular-Linear Correlation (R) of a dataset.
    Returns R, a value between 0.0 (Independent) and 1.0 (Perfectly Correlated).
    Uses a one-pass accumulation strategy for memory efficiency on large datasets.
    """
    # Accumulators in double precision to avoid numerical overflow/errors
    sum_A = 0.0
    sum_C = 0.0
    sum_S = 0.0
    sum_A2 = 0.0
    sum_C2 = 0.0
    sum_S2 = 0.0
    sum_AC = 0.0
    sum_AS = 0.0
    sum_CS = 0.0
    total_pixels = 0

    for batch in dataloader:
        # 1. Data extraction (handles different batch formats)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
            
        x = x.to(device)
        
        if not x.is_complex():
            raise ValueError("Tensor is not complex. If you use a Dual-Real format, convert it first.")

        # 2. Extract Amplitude and Phase components
        # Casting to double for accumulation stability over millions of pixels
        A = torch.abs(x).double() 
        Phi = torch.angle(x).double()
        
        C = torch.cos(Phi)
        S = torch.sin(Phi)

        # 3. Accumulate sums (.item() unloads the GPU memory)
        sum_A += A.sum().item()
        sum_C += C.sum().item()
        sum_S += S.sum().item()
        
        sum_A2 += (A ** 2).sum().item()
        sum_C2 += (C ** 2).sum().item()
        sum_S2 += (S ** 2).sum().item()
        
        sum_AC += (A * C).sum().item()
        sum_AS += (A * S).sum().item()
        sum_CS += (C * S).sum().item()
        
        total_pixels += x.numel()

    if total_pixels == 0:
        return 0.0

    # 4. Calculate Expectations (E[X])
    E_A = sum_A / total_pixels
    E_C = sum_C / total_pixels
    E_S = sum_S / total_pixels
    
    E_A2 = sum_A2 / total_pixels
    E_C2 = sum_C2 / total_pixels
    E_S2 = sum_S2 / total_pixels
    
    E_AC = sum_AC / total_pixels
    E_AS = sum_AS / total_pixels
    E_CS = sum_CS / total_pixels

    # 5. Calculate Variances and Covariances
    var_A = E_A2 - E_A**2
    var_C = E_C2 - E_C**2
    var_S = E_S2 - E_S**2
    
    cov_AC = E_AC - E_A * E_C
    cov_AS = E_AS - E_A * E_S
    cov_CS = E_CS - E_C * E_S

    if var_A < 1e-12 or var_C < 1e-12 or var_S < 1e-12:
        print("Warning: Zero variance detected. Correlation is undefined.")
        return 0.0

    # 6. Calculate Pearson Correlations
    r_xc = cov_AC / math.sqrt(var_A * var_C)
    r_xs = cov_AS / math.sqrt(var_A * var_S)
    r_cs = cov_CS / math.sqrt(var_C * var_S)

    # Ensure correlations are strictly within [-1, 1]
    r_xc = max(min(r_xc, 1.0), -1.0)
    r_xs = max(min(r_xs, 1.0), -1.0)
    r_cs = max(min(r_cs, 1.0), -1.0)

    # 7. Mardia's formula for circular-linear correlation (R^2)
    numerator = r_xc**2 + r_xs**2 - 2 * r_xc * r_xs * r_cs
    denominator = 1 - r_cs**2
    
    if denominator < 1e-12:
        R = 0.0
    else:
        R2 = numerator / denominator
        R = math.sqrt(max(min(R2, 1.0), 0.0))

    print(f"--- Circular-Linear Correlation Statistics ---")
    print(f"Pearson r(Amplitude, cos(Phi)): {r_xc:.4f}")
    print(f"Pearson r(Amplitude, sin(Phi)): {r_xs:.4f}")
    print(f"Pearson r(cos(Phi), sin(Phi)) : {r_cs:.4f}")
    print(f"Circular-Linear Correlation (R): {R:.4f}")

    return R

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            targets = batch[1]
        else:
            inputs = batch
            targets = inputs 
            
        inputs = inputs.to(device)  
        targets = targets.to(device)

        # --- FORWARD & BACKWARD ---
        optimizer.zero_grad()
        _, outputs = model(inputs)
        
        if isinstance(criterion, c_nn.ComplexMSELoss):
            loss = criterion(outputs, inputs)
        else:
            loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        
        if not isinstance(criterion, c_nn.ComplexMSELoss):
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()            

    avg_loss = total_loss / len(loader)
    if isinstance(criterion, c_nn.ComplexMSELoss):
        acc = 0.0  # No accuracy for autoencoders
    else:
        acc = 100 * correct / total
    return avg_loss, acc

def validate_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set without updating gradients.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Crucial to prevent tracking history and saving memory
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                targets = batch[1]
            else:
                inputs = batch
                targets = inputs 
                
            inputs = inputs.to(device)  
            targets = targets.to(device)

            # --- FORWARD PASS ONLY ---
            _, outputs = model(inputs)
            
            if isinstance(criterion, c_nn.ComplexMSELoss):
                loss = criterion(outputs, inputs)
            else:
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            
            if not isinstance(criterion, c_nn.ComplexMSELoss):
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()            

    avg_loss = total_loss / len(loader)
    if isinstance(criterion, c_nn.ComplexMSELoss):
        acc = 0.0  # No accuracy for autoencoders
    else:
        acc = 100 * correct / total
    return avg_loss, acc

# ==============================================================================

def build_config_dict(args):
    transforms_list = []

    if "mstar" in args.dataset.lower():
        transforms_list.append({
            "name": "fftresize", 
            "params": {"size": [128, 128]},
        })
        transforms_list.append({
            "name": "centercrop", 
            "params": {"height": args.image_size, "width": args.image_size}
        })
        transforms_list.append({
            "name": "global_scalar_normalize",
        })
        type = "sar"
        task = "classification"
        has_labels = True
        recompute_statistics = True
        shuffle = True
        num_channels = 1
    elif "mnist" in args.dataset.lower():
        transforms_list.append({"name": "fft2"})
        transforms_list.append({
            "name": "global_scalar_normalize",
        })
        type = "grayscale"
        task = "classification"
        has_labels = True
        recompute_statistics = True
        shuffle = True
        num_channels = 1
    return {
        "task": task,
        "model": {"layer_mode": "complex"},
        "data": {
            "batch_size": args.batch_size,
            "has_labels": has_labels,
            "num_workers": 4,
            "recompute_statistics": recompute_statistics,
            "shuffle": shuffle,
            "supports_full_image_reconstruction": False,
            "test_ratio": 0.1,
            "valid_ratio": 0.1,
            "dataset": {
                "name": args.dataset,       # ex: "MNIST", "Miccai2023"
                "trainpath": args.dataroot, # ex: "../datasets/MNIST"
            },
            "transforms": transforms_list, 
            "type": type,
            "num_channels": num_channels,
        }
    }

# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Feature Extractors (RVNN) for FID.")
    
    # Arguments
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Nom du dataset (ex: MNIST, Miccai2023, MSTAR)")
    parser.add_argument("--dataroot", type=str, default="../datasets", 
                        help="Path to the folder containing the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints/classifiers", help="Folder to save trained models")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes (if classification)")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Target size for resize (MSTAR/SAR). Default=128.")
    
    args = parser.parse_args()
    
    run_name = f"train_{args.dataset.lower()}_extractor"
    wandb.init(
        entity="complex-gen-ai-sar",
        project="cvnn-feature-extractors",
        name=run_name,
        config=vars(args)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Feature Extractor on {device}")
    print(f"Dataset: {args.dataset} | Root: {args.dataroot}")

    cfg = build_config_dict(args)

    loaders = get_dataloaders(cfg, use_cuda=torch.cuda.is_available())
    
    if len(loaders) == 3:
        train_loader, valid_loader, test_loader = loaders
    else:
        train_loader, valid_loader = loaders
        
    print(f"✅ DataLoaders chargés. Train size: {len(train_loader.dataset)}")
        
    rho = compute_dataset_impropriety(train_loader, device='cuda')
    print(f"Le coefficient d'impropriété du dataset est de {rho * 100:.1f}%")

    circular_correlation = compute_dataset_circular_linear_correlation(train_loader, device='cuda')
    print(f"Le coefficient de corrélation circulaire-linéaire du dataset est de {circular_correlation:.4f}")
    
    wandb.config.update({
        "dataset_impropriety_rho": rho,
        "dataset_circular_correlation": circular_correlation
    })
    
    dataset_clean = args.dataset.lower()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if "mnist" in dataset_clean:
        print("🏗️  Model: MnistClassifier (LeNet-style)")
        model = MnistClassifier(num_classes=args.num_classes)
        criterion = nn.CrossEntropyLoss()
        save_name = "mnist_classifier.pt"
            
    elif "sar" in dataset_clean or "mstar" in dataset_clean:
        print("🏗️  Model: MSTARClassifier (LeNet-style)")
        model = MstarClassifier(num_classes=args.num_classes)
        criterion = nn.CrossEntropyLoss()
        save_name = "mstar_classifier.pt"
                
    else:
        print(f"Dataset '{args.dataset}' unknown.")
        wandb.finish() # Ferme W&B si on quitte prématurément
        return

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=5e-6)
    
    save_path = save_dir / save_name
    best_val_loss = float('inf') # Track the best validation loss

    for epoch in range(args.epochs):
        # 1. Training Phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 2. Validation Phase
        val_loss, val_acc = validate_one_epoch(
            model, valid_loader, criterion, device
        )
        scheduler.step(val_loss)
        
        # Print metrics
        status = f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%"
        print(f"   Epoch {epoch+1}/{args.epochs} | {status}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        # 3. Save the model only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved! (Val Loss: {val_loss:.4f})")
            
            wandb.log({"val/best_loss": best_val_loss, "epoch": epoch + 1})

    print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()
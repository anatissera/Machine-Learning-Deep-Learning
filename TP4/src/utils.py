
import cupy as cp
from models import VAE
import numpy as np
from torch.utils.data import DataLoader, Subset

def find_elbow(Ks, Ls, alpha=0.02):
    """
    Ks: lista de K (enteros)
    Ls: lista de inercia correspondiente
    alpha: fracción mínima (p.ej. 0.1 = 10%) de la reducción inicial D1
    
    Devuelve: (bestK, Ds), donde bestK es el K elegido,
    y Ds es el array con los Dk relativos a D1 (en NumPy).
    """

    Ls_cp = cp.array(Ls, dtype=cp.float64)
    
    # Calcular disminuciones Dk = L[k-1] - L[k]
    D_cp = Ls_cp[:-1] - Ls_cp[1:]
    # Normalizar por la primera reducción D1
    D_rel_cp = D_cp / D_cp[0]
    
    # Buscar primer índice i donde D_rel < alpha
    mask = D_rel_cp < alpha
    if cp.any(mask):
        idx = int(cp.argmax(mask))
        bestK = Ks[idx+1]           
    else:
        # Si nunca cae por debajo del umbral, escogemos el ratio máximo (codo viejo)
        idx = int(cp.argmax(cp.abs(cp.diff(D_rel_cp, n=1))))
        bestK = Ks[idx+1]

    Ds = cp.asnumpy(D_rel_cp)
    
    return bestK, Ds


def cv(latent_list, lr_list, hidden_list, num_folds, fold_size, indices_trainval, trainval_ds, device):
    for latent_dim in latent_list:
        for lr in lr_list:
            for hidden_dim in hidden_list:
                cv_losses = []

                for fold in range(num_folds):
                    start = fold * fold_size
                    end = start + fold_size if fold < num_folds - 1 else len(indices_trainval)
                    val_fold_idx   = indices_trainval[start:end]
                    train_fold_idx = np.setdiff1d(indices_trainval, val_fold_idx)

                    train_subset = Subset(trainval_ds, train_fold_idx.tolist())
                    val_subset   = Subset(trainval_ds, val_fold_idx.tolist())

                    train_loader_fold = DataLoader(train_subset, batch_size=128, shuffle=True)
                    val_loader_fold   = DataLoader(val_subset,   batch_size=128, shuffle=False)

                    vae_fold = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

                    train_losses_fold, val_losses_fold = vae_fold.fit(
                        train_loader_fold,
                        val_loader_fold,
                        n_epochs=7,
                        lr=lr,
                        device=device
                    )

                    fold_val_loss = val_losses_fold[-1]
                    cv_losses.append(fold_val_loss)

                avg_cv_loss = np.mean(cv_losses)
                print(f"latent={latent_dim}, hidden={hidden_dim}, lr={lr} → CV_loss={avg_cv_loss:.3f}")

                if avg_cv_loss < best_cv_loss:
                    best_cv_loss = avg_cv_loss
                    best_cfg = (latent_dim, hidden_dim, lr)
                    
    return best_cfg, best_cv_loss
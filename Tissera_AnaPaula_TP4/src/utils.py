
import cupy as cp

def find_elbow(Ks, Ls, alpha=0.02):
    """
    Ks: lista de K (enteros)
    Ls: lista de inercia correspondiente
    alpha: fracción mínima (p.ej. 0.1 = 10%) de la reducción inicial D1
    
    Devuelve: (bestK, Ds_cpu), donde bestK es el K elegido,
    y Ds_cpu es el array con los Dk relativos a D1 (en NumPy).
    """
    # Pasar Ls a CuPy para cálculo GPU
    Ls_cp = cp.array(Ls, dtype=cp.float64)
    # Calcular disminuciones Dk = L[k-1] - L[k]
    D_cp = Ls_cp[:-1] - Ls_cp[1:]
    # Normalizar por la primera reducción D1
    D_rel_cp = D_cp / D_cp[0]
    # Buscar primer índice i donde D_rel < alpha
    # Atención: i corresponde a la disminución entre K=i+1 y K=i+2
    mask = D_rel_cp < alpha
    if cp.any(mask):
        idx = int(cp.argmax(mask))  # el primer True
        bestK = Ks[idx+1]           # corresponde a K = idx+2, pero usamos idx+1 (0‑based)
    else:
        # Si nunca cae por debajo del umbral, escogemos el máximo ratio (viejo codo)
        idx = int(cp.argmax(cp.abs(cp.diff(D_rel_cp, n=1))))
        bestK = Ks[idx+1]
    # Traer a NumPy solo para inspección (opcional)
    Ds_cpu = cp.asnumpy(D_rel_cp)
    return bestK, Ds_cpu
#
#
#                .    o8o  oooo   o8o      .                                          
#              .o8    `"'  `888   `"'    .o8                                          
#  oooo  oooo  .o888oo oooo   888  oooo  .o888oo oooo    ooo     oo.ooooo.  oooo    ooo 
#  `888  `888    888   `888   888  `888    888    `88.  .8'       888' `88b  `88.  .8'  
#   888   888    888    888   888   888    888     `88..8'        888   888   `88..8'   
#   888   888    888 .  888   888   888    888 .    `888'    .o.  888   888    `888'    
#   `V88V"V8P'   "888" o888o o888o o888o   "888"     .8'     Y8P  888bod8P'     .8'     
#                                              .o..P'           888       .o..P'      
#                                              `Y8P'           o888o      `Y8P'       
#
#                                                                                     

import pandas as pd
import numpy  as np

# ------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------
def _embed_indices(N, d, tau):
    """Matriz (M x d) de índices para embedding con retardo tau.
       M = N - (d-1)*tau; si M<=0 retorna None."""
    M = int(N - (d - 1) * tau)
    if M <= 0:
        return None
    base  = np.arange(M)[:, None]           # (M x 1)
    steps = (np.arange(d)[None, :] * tau)   # (1 x d)
    return base + steps                     # (M x d)

def _shannon_entropy(p):
    """Entropía de Shannon (nats). Ignora p<=0 por convención."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())

def _quantize_dispersion_minmax(x, c):
    """Min–max → [0,1]; cuantiza a 1..c con floor(c*Xi+0.5) y clamp."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=int)
    xmin = np.nanmin(x); xmax = np.nanmax(x)
    if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or xmax == xmin:
        Xi = np.zeros_like(x)
    else:
        Xi = (x - xmin) / (xmax - xmin)
    Yi = np.floor(c * Xi + 0.5).astype(int)  # cuantización
    Yi[Yi < 1] = 1
    Yi[Yi > c] = c
    return Yi

def _dispersion_histogram(x, d, tau, c):
    """
    Histograma de patrones de dispersión (K=c^d bins) con codificación
    base-c CONSISTENTE: pesos ascendentes [1, c, c^2, ...].
    Devuelve (p, K) con p normalizado.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or d < 1 or tau < 1 or c < 2:
        return None, None

    idx = _embed_indices(x.size, d, tau)
    if idx is None:
        return None, None

    # 1) cuantización 1..c
    Y = _quantize_dispersion_minmax(x, c)

    # 2) embedding y 3) codificación base-c (Yi-1)
    emb   = Y[idx]                             # (M x d) con valores 1..c
    K     = int(c ** d)
    cpows = (c ** np.arange(d)).astype(int)    # [1, c, c^2, ...]  (clave p/consistencia)
    codes = ((emb - 1) * cpows).sum(axis=1)    # 0..K-1

    # 4) histograma → p
    counts = np.bincount(codes, minlength=K)
    total  = counts.sum()
    if total == 0:
        return None, None
    p = counts / total
    return p.astype(float), K

# ------------------------------------------------------------
# CResidual-Dispersion Entropy (CRDE) — normalizada [0,1]
# Firma EXACTA de la plantilla
# ------------------------------------------------------------
def entropy_dispersion(x, d, tau, c):
    """
    Dispersion Entropy (Shannon) normalizada en [0,1].
    """
    # usa SIEMPRE el mismo histograma/codificación que el resto
    p, K = _dispersion_histogram(x, d, tau, c)
    if p is None:
        return np.nan
    H = _shannon_entropy(p)
    
    return float(H / np.log(K)) if K > 1 else np.nan

def _entropy_dispersion_cr(x, d, tau, c):
    """
    CR Dispersion Entropy (residual), normalizada en [0,1] (versión opcional).
    """
    p, K = _dispersion_histogram(x, d, tau, c)
    if p is None:
        return np.nan
    S = np.cumsum(p[::-1])[::-1]
    Spos = S[S > 0]
    H = float(-(Spos * np.log(Spos)).sum())
    k = np.arange(K)
    Sunif = (K - k) / K
    Hmax = float(-(Sunif[Sunif > 0] * np.log(Sunif[Sunif > 0])).sum())
    return float(H / Hmax) if Hmax > 0 else np.nan


# ------------------------------------------------------------
# Permutation Entropy (PE) — normalizada [0,1]
# 
# ------------------------------------------------------------
def entropy_permuta(x, m, tau):
    """
    Permutation Entropy normalizada en [0,1] sobre la serie x.
    Pasos: embedding (m,tau) → patrón ordinal estable → histograma (m!) →
           Shannon / log(m!)
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or m < 2 or tau < 1:
        return np.nan

    idx = _embed_indices(x.size, m, tau)
    if idx is None:
        return np.nan

    emb = x[idx]                       # (M x m)
    M   = emb.shape[0]
    codes = np.empty(M, dtype=int)

    # tie-break estable: por valores y luego por índice
    idx_local = np.arange(m)
    for i in range(M):
        row  = emb[i]
        perm = np.lexsort((idx_local, row))  # ascendente y estable

        # Código tipo Lehmer (simple)
        code = 0
        for a in range(m):
            smaller = 0
            for b in range(a+1, m):
                if perm[b] < perm[a]:
                    smaller += 1
            # multiplicador = (m-1-a)!
            mult = 1
            for t in range(2, m - a):
                mult *= t
            code += smaller * mult
        codes[i] = code

    # m! bins
    K = 1
    for t in range(2, m+1):
        K *= t

    counts = np.bincount(codes, minlength=K)
    total  = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total

    H = _shannon_entropy(p)
    return float(H / np.log(K)) if K > 1 else np.nan

def entropy_mde(x, d, tau, c, S_max):
    """
    Multi-Scale Dispersion Entropy (MDE)
    x: serie temporal (array 1D)
    d: dimension embedding
    tau: factor de retardo
    c: número de símbolos
    S_max: número máximo de escalas
    Devuelve: array de MDE para escalas 1 a S_max
    """
    N = len(x)
    mde_values = np.zeros(S_max)

    for scale in range(1, S_max + 1):
        # Paso 1: Dividir en ventanas de tamaño scale
        window_size = scale
        num_windows = N // window_size

        if num_windows == 0:
            mde_values[scale-1] = np.nan
            continue

        # Calcular promedio de cada ventana
        y = np.zeros(num_windows)
        for j in range(num_windows):
            start = j * window_size
            end = (j + 1) * window_size
            y[j] = np.mean(x[start:end])

        # Paso 2: Calcular entropía de dispersión para y
        mde_values[scale-1] = entropy_dispersion(y, d, tau, c)

    return mde_values

def entropy_emde(x, d, tau, c, S_max):
    """
    Enhanced Multi-Scale Dispersion Entropy (eMDE)
    x: serie temporal (array 1D)
    d: dimension embedding
    tau: factor de retardo
    c: número de símbolos
    S_max: número máximo de escalas
    Devuelve: array de eMDE para escalas 1 a S_max
    """
    N = len(x)
    emde_values = np.zeros(S_max)

    # Escala 1: entropía de la serie original
    emde_values[0] = entropy_dispersion(x, d, tau, c)

    for scale in range(2, S_max + 1):
        # Paso 1: Generar sub-series
        avg_entropy = 0.0
        for k in range(1, scale + 1):
            # Sub-serie u_k = x[k:N]
            u_k = x[k-1:]  # Ajuste de índice para 0-based

            # Paso 2: Segmentar sub-serie
            window_size = scale
            num_windows = len(u_k) // window_size

            if num_windows == 0:
                continue

            z = np.zeros(num_windows)
            for j in range(num_windows):
                start = j * window_size
                end = (j + 1) * window_size
                z[j] = np.mean(u_k[start:end])

            # Calcular entropía para esta sub-serie
            E_k = entropy_dispersion(z, d, tau, c)
            avg_entropy += E_k

        # Promedio de entropías para esta escala
        emde_values[scale-1] = avg_entropy / scale

    return emde_values

def entropy_mpe(x, m, tau, S_max):
    """
    Multi-Scale Permutation Entropy (MPE)
    x: serie temporal (array 1D)
    m: dimension embedding
    tau: factor de retardo
    S_max: número máximo de escalas
    Devuelve: array de MPE para escalas 1 a S_max
    """
    N = len(x)
    mpe_values = np.zeros(S_max)

    for scale in range(1, S_max + 1):
        # Paso 1: Dividir en ventanas de tamaño scale
        window_size = scale
        num_windows = N // window_size

        if num_windows == 0:
            mpe_values[scale-1] = np.nan
            continue

        # Calcular promedio de cada ventana
        y = np.zeros(num_windows)
        for j in range(num_windows):
            start = j * window_size
            end = (j + 1) * window_size
            y[j] = np.mean(x[start:end])

        # Paso 2: Calcular entropía de permutación para y
        mpe_values[scale-1] = entropy_permuta(y, m, tau)

    return mpe_values

def entropy_empe(x, m, tau, S_max):
    """
    Enhanced Multi-Scale Permutation Entropy (eMPE)
    x: serie temporal (array 1D)
    m: dimension embedding
    tau: factor de retardo
    S_max: número máximo de escalas
    Devuelve: array de eMPE para escalas 1 a S_max
    """
    N = len(x)
    empe_values = np.zeros(S_max)

    # Escala 1: entropía de la serie original
    empe_values[0] = entropy_permuta(x, m, tau)

    for scale in range(2, S_max + 1):
        # Paso 1: Generar sub-series
        avg_entropy = 0.0
        for k in range(1, scale + 1):
            # Sub-serie u_k = x[k:N]
            u_k = x[k-1:]  # Ajuste de índice para 0-based

            # Paso 2: Segmentar sub-serie
            window_size = scale
            num_windows = len(u_k) // window_size

            if num_windows == 0:
                continue

            z = np.zeros(num_windows)
            for j in range(num_windows):
                start = j * window_size
                end = (j + 1) * window_size
                z[j] = np.mean(u_k[start:end])

            # Calcular entropía para esta sub-serie
            E_k = entropy_permuta(z, m, tau)
            avg_entropy += E_k

        # Promedio de entropías para esta escala
        empe_values[scale-1] = avg_entropy / scale

    return empe_values

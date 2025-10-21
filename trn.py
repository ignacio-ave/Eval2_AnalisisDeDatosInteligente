
#
#      .                                                   
#    .o8                                                   
#  .o888oo oooo d8b ooo. .oo.       oo.ooooo.  oooo    ooo 
#    888   `888""8P `888P"Y88b       888' `88b  `88.  .8'  
#    888    888      888   888       888   888   `88..8'   
#    888 .  888      888   888  .o.  888   888    `888'    
#    "888" d888b    o888o o888o Y8P  888bod8P'     .8'     
#                                    888       .o..P'      
#                                   o888o      `Y8P'       
#                                                        
#



import os
import numpy as np
import pandas as pd
from typing import Tuple

# Constantes de rutas
DATA_DIR = "data"
CONF_DIR = "config"
CONF_TRAIN = os.path.join(CONF_DIR, "conf_train.csv")

def cargar_datos_trn(ruta_X: str, ruta_y: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga datos de características y etiquetas desde archivos CSV

    Args:
        ruta_X: Ruta al archivo dClases.csv
        ruta_y: Ruta al archivo dLabel.csv

    Returns:
        X: Matriz de características (N x D)
        y: Matriz de etiquetas (N x K)

    Raises:
        ValueError: Si las dimensiones no coinciden o hay valores no finitos
    """
    X = pd.read_csv(ruta_X, header=None).values.astype(float)
    y = pd.read_csv(ruta_y, header=None).values.astype(float)

    # Validaciones (RNF-001)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Número de filas diferente en X ({X.shape[0]}) e y ({y.shape[0]})")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contiene valores no finitos")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene valores no finitos")

    return X, y

def reordenar_aleatoriamente(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reordena aleatoriamente X y y de manera sincronizada

    Args:
        X: Matriz de características
        y: Matriz de etiquetas
        seed: Semilla para reproducibilidad

    Returns:
        X_shuffled: Matriz de características reordenada
        y_shuffled: Matriz de etiquetas reordenada
    """
    # Validación (RNF-002, RNF-003)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y deben tener el mismo número de filas")

    np.random.seed(seed)
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

def normalizar_zscore(X: np.ndarray) -> np.ndarray:
    """
    Normaliza características usando Z-score: x = (x - mean(x)) / std(x)

    Args:
        X: Matriz de características

    Returns:
        X_norm: Matriz normalizada

    Raises:
        ValueError: Si hay división por cero
    """
    # Validación (RNF-004)
    if X.size == 0:
        return X

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    if np.any(std == 0):
        raise ValueError("Desviación estándar cero detectada - no se puede normalizar")

    return (X - mean) / std

def dividir_train_test(X_norm: np.ndarray, y_shuffled: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide datos en conjuntos de entrenamiento y prueba

    Args:
        X_norm: Matriz de características normalizada
        y_shuffled: Matriz de etiquetas reordenada
        p: Porcentaje para entrenamiento (0.65 < p < 0.80)

    Returns:
        X_train, X_test, y_train, y_test

    Raises:
        ValueError: Si p no está en el rango válido
    """
    # Validación (RNF-005)
    if not (0.65 <= p <= 0.80):
        raise ValueError("p debe estar entre 0.65 y 0.80")

    N = X_norm.shape[0]
    L = int(np.round(N * p))
    L = max(1, min(L, N - 1))  # Asegurar al menos una muestra en cada conjunto

    return X_norm[:L], X_norm[L:], y_shuffled[:L], y_shuffled[L:]

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Calcula la función softmax de manera numéricamente estable

    Args:
        z: Matriz de scores lineales

    Returns:
        Matriz de probabilidades
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def entrenar_mgd(X_train: np.ndarray, y_train: np.ndarray, max_iter: int, mu: float, beta: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Entrena modelo usando descenso de gradiente con momentum (mGD)
    """
    print("\n[ETAPA] Entrenamiento con mGD")
    print(f"[INFO] X_train.shape = {X_train.shape}")
    print(f"[INFO] y_train.shape = {y_train.shape}")
    print(f"[INFO] max_iter = {max_iter}, mu = {mu}, beta = {beta}")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train y y_train deben tener el mismo número de muestras")

    D = X_train.shape[1]
    K = y_train.shape[1]

    print(f"[INFO] D (features) = {D}, K (clases) = {K}")

    # Inicializar pesos
    W = np.random.randn(D, K) * 0.01  # W inicial sin bias
    print(f"[INFO] W inicial (sin bias) shape = {W.shape}")

    # Agregar bias a entrada y pesos
    Xb = np.hstack([X_train, np.ones((X_train.shape[0], 1))])     # Bias como columna extra en X
    W = np.vstack([W, np.zeros((1, K))])                           # Bias como fila extra en W
    print(f"[INFO] Xb shape (con bias) = {Xb.shape}")
    print(f"[INFO] W shape (con bias) = {W.shape}")

    V = np.zeros_like(W)
    J = np.zeros(max_iter)
    eps = 1e-15

    for it in range(max_iter):
        z = Xb @ W
        yhat = softmax(z)

        J[it] = -np.mean(np.sum(y_train * np.log(yhat + eps), axis=1))

        grad = (Xb.T @ (yhat - y_train)) / X_train.shape[0]
        V = beta * V + mu * grad
        W = W - V
        if it == 0 or (it + 1) % (max_iter // 5) == 0 or it == max_iter - 1:
            preds = np.argmax(yhat, axis=1)
            clases, counts = np.unique(preds, return_counts=True)
            print(f"[DEBUG] Predicciones por clase: {dict(zip(clases, counts))}")
    
            print(f"[INFO] Iteración {it+1}/{max_iter} → Costo J = {J[it]:.6f}")

    print("[ok] Entrenamiento finalizado.\n")
    return W, J


def guardar_pesos_y_costo(W: np.ndarray, J: np.ndarray, ruta_pesos: str, ruta_costo: str) -> None:
    """
    Guarda pesos y costos en archivos CSV

    Args:
        W: Matriz de pesos
        J: Vector de costos
        ruta_pesos: Ruta para guardar pesos
        ruta_costo: Ruta para guardar costos

    Raises:
        ValueError: Si las dimensiones no son válidas
    """
    # Validación (RNF-008)
    if W.ndim != 2:
        raise ValueError("W debe ser una matriz 2D")
    if J.ndim != 1:
        raise ValueError("J debe ser un vector 1D")

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(W).to_csv(ruta_pesos, index=False, header=False)
    pd.DataFrame(J).to_csv(ruta_costo, index=False, header=False)

def cargar_configuracion(ruta: str = CONF_TRAIN) -> Tuple[int, float, float]:
    """
    Carga configuración desde archivo conf_train.csv

    Args:
        ruta: Ruta al archivo de configuración

    Returns:
        max_iter: Número máximo de iteraciones
        mu: Tasa de aprendizaje
        p: Porcentaje para entrenamiento

    Raises:
        ValueError: Si los valores no son válidos
    """
    conf = pd.read_csv(ruta, header=None).values

    # Validaciones (RNF-009)
    max_iter = int(conf[0, 0])
    mu = float(conf[1, 0])
    p = float(conf[2, 0]) / 100.0 if conf[2, 0] > 1 else conf[2, 0]

    if not (0 < mu < 1):
        raise ValueError("mu debe estar entre 0 y 1")
    if not (0.65 <= p <= 0.80):
        raise ValueError("p debe estar entre 0.65 y 0.80")

    return max_iter, mu, p

def trn():
    """
    Función principal para ejecutar el proceso de entrenamiento
    """
    # Cargar configuración
    max_iter, mu, p = cargar_configuracion()

    # Cargar datos
    X, y = cargar_datos_trn(os.path.join(DATA_DIR, "dClases.csv"), 
                       os.path.join(DATA_DIR, "dLabel.csv"))

    # Reordenar aleatoriamente
    X_shuffled, y_shuffled = reordenar_aleatoriamente(X, y)

    # Normalizar
    X_norm = normalizar_zscore(X_shuffled)

    # Dividir en train/test
    X_train, X_test, y_train, y_test = dividir_train_test(X_norm, y_shuffled, p)

    # Guardar conjuntos de entrenamiento y prueba
    pd.DataFrame(X_train).to_csv(os.path.join(DATA_DIR, "dtrn.csv"), index=False, header=False)
    pd.DataFrame(y_train).to_csv(os.path.join(DATA_DIR, "dtrn_label.csv"), index=False, header=False)
    pd.DataFrame(X_test).to_csv(os.path.join(DATA_DIR, "dtst.csv"), index=False, header=False)
    pd.DataFrame(y_test).to_csv(os.path.join(DATA_DIR, "dtst_label.csv"), index=False, header=False)

    # Entrenar modelo
    W, J = entrenar_mgd(X_train, y_train, max_iter, mu)

    # Guardar resultados
    guardar_pesos_y_costo(W, J,
                         os.path.join(DATA_DIR, "pesos.csv"),
                         os.path.join(DATA_DIR, "costo.csv"))

    print("[ok] Entrenamiento finalizado. Archivos guardados en data/")






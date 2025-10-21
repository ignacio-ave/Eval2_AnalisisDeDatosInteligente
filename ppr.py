#                                                           
#                                                          
#  oo.ooooo.  oo.ooooo.  oooo d8b     oo.ooooo.  oooo    ooo 
#   888' `88b  888' `88b `888""8P      888' `88b  `88.  .8'  
#   888   888  888   888  888          888   888   `88..8'   
#   888   888  888   888  888     .o.  888   888    `888'    
#   888bod8P'  888bod8P' d888b    Y8P  888bod8P'     .8'     
#   888        888                     888       .o..P'      
#  o888o      o888o                   o888o      `Y8P'       
#                                                          
#

import os
import numpy as np
import pandas as pd
from utility import (entropy_dispersion, entropy_permuta,
                    entropy_mde, entropy_emde, entropy_mpe, entropy_empe)
# Constantes de rutas
DATA_DIR = "data"
CONF_DIR = "config"
CONF_TEMP = os.path.join(CONF_DIR, "conf_temp.csv")
CONF_OPTIMO = os.path.join(CONF_DIR, "conf_optimo.csv")

def cargar_datos_ppr(rutas_classes):
    """
    Carga los datos de las 4 clases desde archivos CSV
    Returns:
        numpy.ndarray: Matriz X de tamaño (120000, 4)
    """
    matrices = []
    for ruta in rutas_classes:
        print(f"[INFO] Cargando archivo: {ruta}")
        df = pd.read_csv(ruta, header=None)
        print(f"[DEBUG] {ruta} → shape = {df.shape}")
        matriz = df.values.astype(float)
        if matriz.ndim == 1:
            matriz = matriz[:, None]
        # Seleccionar solo una columna por clase (ej. columna 0)
        matrices.append(matriz[:, [0]])

    # Validación de dimensiones
    for i, matriz in enumerate(matrices):
        if matriz.shape[0] != 120000:
            print(f"[ERROR] Clase #{i+1} no tiene 120000 filas → tiene {matriz.shape[0]}")
        if np.any(~np.isfinite(matriz)):
            print(f"[ERROR] Clase #{i+1} contiene NaN o infinitos.")

    X = np.hstack(matrices)
    print(f"[INFO] X final → shape = {X.shape} (esperado: (120000, 4))")
    return X


def aplicar_derivada_discreta(X, metodo="backward", mantener_longitud=True):
    """
    Calcula la derivada discreta (diferencia finita) de la matriz X.
    Opciones disponibles:
        - "backward":  ẋ[n] = x[n] - x[n-1]
        - "forward" :  ẋ[n] = x[n+1] - x[n]
        - "central" :  ẋ[n] = (x[n+1] - x[n-1]) / 2
    La opción correcta para este proyecto es "backward" con mantener_longitud=True.
    Returns:
        numpy.ndarray: X_der ∈ ℜ^(N, M)
    """
    import numpy as np

    print(f"[INFO] aplicando derivada → X.shape = {X.shape}")
    N, M = X.shape

    if X.shape != (120000, 4):
        print(f"[WARN] X no tiene forma esperada (120000, 4) → {X.shape}")

    X = np.asarray(X, dtype=float)
    X_der = np.empty_like(X)

    if metodo == "backward":
        # Derivada backward con longitud preservada
        X_der[0, :] = 0.0
        X_der[1:, :] = X[1:, :] - X[:-1, :]
        print(f"[INFO] Método: backward (correcto) — mantiene longitud N={N}")

    elif metodo == "forward":
        X_der[:-1, :] = X[1:, :] - X[:-1, :]
        X_der[-1, :] = 0.0 if mantener_longitud else np.nan
        print(f"[INFO] Método: forward — {'mantiene' if mantener_longitud else 'reduce'} longitud")

    elif metodo == "central":
        X_der[1:-1, :] = (X[2:, :] - X[:-2, :]) / 2.0
        if mantener_longitud:
            X_der[0, :] = X_der[1, :]
            X_der[-1, :] = X_der[-2, :]
        else:
            X_der = X_der[1:-1, :]
        print(f"[INFO] Método: central — {'mantiene' if mantener_longitud else 'reduce'} longitud")

    else:
        raise ValueError("Método no válido: use 'backward', 'forward' o 'central'.")

    print(f"[INFO] X_der.shape = {X_der.shape} (esperado: ({N}, {M}))")
    return X_der

"""def segmentar_muestras(X_diff, lF):
    filas = X_diff.shape[0]
    print(f"[INFO] segmentar_muestras: X_diff.shape = {X_diff.shape}, lF = {lF}")
    resto = filas % lF
    print(f"[INFO]  filas % lF = {filas} % {lF} = {resto}")

    if resto != 0:
        print(f"[WARN] {filas} no divisible por {lF}, se recortarán {resto} filas.")
        X_diff = X_diff[:filas - resto, :]

    n_segmentos = X_diff.shape[0] // lF
    print(f"[INFO] Se crearán {n_segmentos} segmentos de tamaño {lF}")
    segmentos = []
    for i in range(n_segmentos):
        start = i * lF
        end = (i + 1) * lF
        segmentos.append(X_diff[start:end, :])
    return segmentos"""
    
def segmentar_muestras(X_der, lF):
    """
    Segmenta la matriz completa (N, 4) de manera uniforme por filas.
    Cada segmento incluye las 4 clases simultáneamente.
    """
    N, M = X_der.shape
    print(f"[INFO] segmentar_muestras: X_der.shape = {X_der.shape}, lF = {lF}")
    resto = N % lF
    if resto != 0:
        print(f"[WARN] {resto} filas no divisibles por lF, se recortan.")
        X_der = X_der[:N - resto, :]

    n_segmentos = X_der.shape[0] // lF
    print(f"[INFO] Se crearán {n_segmentos} segmentos de tamaño {lF}")

    segmentos = [X_der[i*lF:(i+1)*lF, :] for i in range(n_segmentos)]
    print(f"[INFO] Total de segmentos generados = {len(segmentos)}")
    return segmentos


def calcular_entropias(segmentos, tipo_entropia, d, tau, c, S_max):
    """
    Calcula entropías multi-escala para los segmentos
    Args:
        segmentos (list): Lista de segmentos
        tipo_entropia (int): 1=MDE, 2=eMDE, 3=MPE, 4=eMPE
        d (int): Dimensión embedding
        tau (int): Factor de retardo
        c (int): Número de símbolos (solo para MDE/eMDE)
        S_max (int): Número máximo de escalas
    Returns:
        pandas.DataFrame: DataFrame con las entropías calculadas
    """
    # Validar parámetros (RNF-008)
    if not all(isinstance(p, int) and p > 0 for p in [d, tau, c, S_max]):
        raise ValueError("Todos los parámetros deben ser enteros positivos")
    features = []
    # Función para calcular entropías según el tipo
    def calcular_entropia_serie(serie):
        if tipo_entropia == 1:
            return entropy_mde(serie, d, tau, c, S_max)
        elif tipo_entropia == 2:
            return entropy_emde(serie, d, tau, c, S_max)
        elif tipo_entropia == 3:
            return entropy_mpe(serie, d, tau, S_max)
        elif tipo_entropia == 4:
            return entropy_empe(serie, d, tau, S_max)
        else:
            raise ValueError("tipo_entropia debe ser 1-4")
    # Procesar cada segmento y cada clase
    for segmento in segmentos:
        for clase in range(4):  # 4 clases
            serie = segmento[:, clase]
            entropias = calcular_entropia_serie(serie)
            features.append(entropias)
    return pd.DataFrame(features)

def generar_etiquetas(n_muestras_por_clase):
    etiquetas = []
    for i in range(4):
        etiquetas.extend([i] * n_muestras_por_clase)
    # Convertir a one-hot encoding
    one_hot = np.eye(4)[etiquetas]
    return pd.DataFrame(one_hot)

def guardar_configuracion(tipo_entropia, lF, d, tau, c, S_max):
    """
    Guarda la configuración en conf_temp.csv
    Args:
        tipo_entropia (int): 1-4
        lF (int): Longitud del segmento
        d (int): Dimensión embedding
        tau (int): Factor de retardo
        c (int): Número de símbolos
        S_max (int): Número máximo de escalas
    """
    # Validar parámetros (RNF-010)
    if not all(isinstance(p, int) and p > 0 for p in [tipo_entropia, lF, d, tau, c, S_max]):
        raise ValueError("Todos los parámetros deben ser enteros positivos")
    if tipo_entropia not in [1, 2, 3, 4]:
        raise ValueError("tipo_entropia debe ser 1-4")
    config = [tipo_entropia, lF, d, tau, c, S_max]
    pd.DataFrame(config).to_csv(CONF_TEMP, index=False, header=False)

def procesar_datos(tipo_entropia, lF, d, tau, c, S_max):
    print("\n[ETAPA] Iniciando preprocesamiento completo...\n")
    rutas = [os.path.join(DATA_DIR, f"class{i+1}.csv") for i in range(4)]
    X = cargar_datos_ppr(rutas)
    X_diff = aplicar_derivada_discreta(X)
    segmentos = segmentar_muestras(X_diff, lF)
    print(f"[INFO] Total de segmentos generados = {len(segmentos)}")
    features = calcular_entropias(segmentos, tipo_entropia, d, tau, c, S_max)
    n_muestras = len(segmentos) * 4
    labels = generar_etiquetas(n_muestras // 4)
    guardar_configuracion(tipo_entropia, lF, d, tau, c, S_max)
    print("[INFO] Preprocesamiento finalizado exitosamente.\n")
    return features, labels


def guardar_resultados(features, labels):
    """
    Guarda los resultados en los archivos correspondientes
    Args:
        features (pandas.DataFrame): DataFrame con características
        labels (pandas.DataFrame): DataFrame con etiquetas
    """
    features.to_csv(os.path.join(DATA_DIR, "dClases.csv"), index=False, header=False)
    labels.to_csv(os.path.join(DATA_DIR, "dLabel.csv"), index=False, header=False)

def ppr():
    # leer conf
    conf = pd.read_csv(CONF_OPTIMO, header=None).values.flatten()
    tipo_entropia, lF, d, tau, c, S_max = map(int, conf)
    # procesar datos
    features, labels = procesar_datos(tipo_entropia, lF, d, tau, c, S_max)
    # guardar resultados
    guardar_resultados(features, labels)

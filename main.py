"""
main.py - Orquestador para búsqueda de parámetros y ejecución con configuración óptima

Este script permite ejecutar el flujo completo de preprocesamiento, entrenamiento y
evaluación de un modelo de clasificación multiclase basado en entropías
multi‑escala y regresión softmax.  Cuenta con dos modos de funcionamiento:

1. **Búsqueda de parámetros** (`--mode search`):
   Realiza una búsqueda aleatoria sobre combinaciones de hiperparámetros de las
   entropías multi‑escala.  Para cada combinación de parámetros se ejecuta el
   preprocesamiento, el entrenamiento y la evaluación completa, guardando el
   macro F1‑score resultante.  Al finalizar, se ordenan los resultados por
   desempeño descendente, se guarda el leaderboard en
   ``config/param_leaderboard.csv`` y se actualiza ``config/conf_optimo.csv``
   con la mejor combinación encontrada.

2. **Ejecución con parámetros óptimos** (`--mode run`):
   Utiliza la configuración actual en ``config/conf_optimo.csv`` para
   preprocesar los datos (con el modo por defecto definido en ``ppr.py``),
   entrenar el modelo y evaluar su rendimiento.  Los archivos de salida
   (dClases.csv, dLabel.csv, dtrn.csv, dtrn_label.csv, dtst.csv,
   dtst_label.csv, pesos.csv, costo.csv, cmatriz.csv, fscores.csv) se
   generan en la carpeta ``data/``.

El objetivo es proporcionar una herramienta sencilla pero potente para
explorar distintas configuraciones de parámetros sin modificar manualmente
los scripts de preprocesamiento o entrenamiento.  Todas las operaciones se
basan en Numpy y Pandas, respetando las buenas prácticas de modularidad y
sin utilizar bibliotecas externas de optimización.

"""

import argparse
import itertools
import json
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

# Importar módulos del pipeline
import ppr
import trn
import test

# Importar funciones de entropía desde utility para cálculo directo
from utility import (
    entropy_mde,
    entropy_emde,
    entropy_mpe,
    entropy_empe,
)


def evaluate_configuration(
    tipo_entropia: int,
    lF: int,
    d: int,
    tau: int,
    c: int,
    S_max: int,
    mode: str = "agregado",
    normalizar: bool = True,
) -> float:
    """
    Ejecuta el proceso completo para una combinación de parámetros y devuelve
    el macro F1‑score obtenido en el conjunto de prueba.

    Parameters
    ----------
    tipo_entropia : int
        Tipo de entropía: 1=MDE, 2=eMDE, 3=MPE, 4=eMPE.
    lF : int
        Longitud del segmento (1000, 1200 o 1600).
    d : int
        Dimensión de embedding (para MDE/eMDE o m en MPE/eMPE).
    tau : int
        Retardo de embedding.
    c : int
        Número de símbolos para MDE/eMDE.
    S_max : int
        Número máximo de escalas.
    mode : str, optional
        Modo de preprocesamiento: "agregado" promedia las entropías de las
        cuatro réplicas de cada clase; "replicas" trata cada réplica de forma
        independiente.  El modo agregado suele ser más rápido y estable para
        búsqueda de hiperparámetros.
    normalizar : bool, optional
        Si True, aplica normalización min–max a las series antes de la derivada.

    Returns
    -------
    float
        Macro F1‑score de la evaluación (promedio del F1 de las 4 clases).
    """
    """
    Para evitar depender de funciones internas de ``ppr``, esta función
    calcula directamente las características y etiquetas utilizando los
    parámetros suministrados.  Se soportan dos modos de preprocesamiento:

    - ``agregado``: se promedian las entropías de las cuatro réplicas
      de cada clase para cada segmento.
    - ``replicas``: trata cada réplica de forma independiente; cada
      segmento de cada réplica se considera una muestra con la etiqueta
      correspondiente a su clase.

    El algoritmo sigue los pasos descritos en las instrucciones: derivada
    backward preservando longitud, segmentación no solapada y cálculo de
    entropías multi‑escala mediante las funciones de ``utility``.
    """
    # Función de entropía según tipo
    def calcular_entropia_segmento(signal: np.ndarray) -> np.ndarray:
        if tipo_entropia == 1:
            return entropy_mde(signal, d, tau, c, S_max)
        elif tipo_entropia == 2:
            return entropy_emde(signal, d, tau, c, S_max)
        elif tipo_entropia == 3:
            return entropy_mpe(signal, d, tau, S_max)
        elif tipo_entropia == 4:
            return entropy_empe(signal, d, tau, S_max)
        else:
            raise ValueError("tipo_entropia debe ser 1-4")

    # Cargar datos de las cuatro clases; cada archivo contiene 4 réplicas
    rutas = [os.path.join(ppr.DATA_DIR, f"class{i+1}.csv") for i in range(4)]
    clases: List[np.ndarray] = []  # Lista de matrices (N, 4)
    for ruta in rutas:
        df = pd.read_csv(ruta, header=None)
        matriz = df.values.astype(float)
        if matriz.ndim == 1:
            matriz = matriz[:, None]
        # Normalizar cada réplica si se indica
        if normalizar:
            # Min–max por columna
            mn = np.nanmin(matriz, axis=0)
            mx = np.nanmax(matriz, axis=0)
            rng = mx - mn
            rng[rng == 0] = 1.0
            matriz_norm = (matriz - mn) / rng
        else:
            matriz_norm = matriz
        clases.append(matriz_norm)

    features: List[np.ndarray] = []
    labels: List[int] = []

    if mode == "replicas":
        # Cada réplica individual
        for class_idx, X in enumerate(clases):
            N, R = X.shape
            # Derivada backward
            deriv = np.empty_like(X)
            deriv[0, :] = 0.0
            deriv[1:, :] = X[1:, :] - X[:-1, :]
            # Número de segmentos
            n_seg = N // lF
            for rep_idx in range(R):
                signal = deriv[:, rep_idx]
                # Recortar para que sea divisible
                usable_len = n_seg * lF
                signal = signal[:usable_len]
                # Segmentar y calcular entropía para cada segmento
                for seg_idx in range(n_seg):
                    seg = signal[seg_idx * lF : (seg_idx + 1) * lF]
                    ent = calcular_entropia_segmento(seg)
                    features.append(ent)
                    labels.append(class_idx)

    elif mode == "agregado":
        # Promediar entropías de las réplicas por clase
        for class_idx, X in enumerate(clases):
            N, R = X.shape
            deriv = np.empty_like(X)
            deriv[0, :] = 0.0
            deriv[1:, :] = X[1:, :] - X[:-1, :]
            n_seg = N // lF
            for seg_idx in range(n_seg):
                entropias_reps: List[np.ndarray] = []
                for rep_idx in range(R):
                    seg = deriv[seg_idx * lF : (seg_idx + 1) * lF, rep_idx]
                    ent = calcular_entropia_segmento(seg)
                    entropias_reps.append(ent)
                # Promediar las entropías de las 4 réplicas
                avg_ent = np.mean(np.vstack(entropias_reps), axis=0)
                features.append(avg_ent)
                labels.append(class_idx)
    else:
        raise ValueError(f"Modo desconocido: {mode}")

    # Convertir a DataFrame y one‑hot labels
    features_df = pd.DataFrame(features)
    labels_onehot = np.eye(4)[labels]
    labels_df = pd.DataFrame(labels_onehot)
    # Guardar datasets para entrenamiento/prueba
    ppr.guardar_resultados(features_df, labels_df)
    # Guardar configuración actual en conf_temp.csv.  Convertimos
    # los parámetros a Python int para evitar problemas con tipos
    # numpy.int64 que no pasan la validación de ppr.guardar_configuracion.
    ppr.guardar_configuracion(
        int(tipo_entropia), int(lF), int(d), int(tau), int(c), int(S_max)
    )
    # Entrenar y evaluar
    trn.trn()
    test.test()
    # Leer F‑scores y calcular macro F1
    fscores_path = os.path.join("data", "fscores.csv")
    fs = pd.read_csv(fscores_path, header=None).values.flatten()
    macro_f1 = float(np.mean(fs))
    return macro_f1


def search_params(
    iterations: int = 5,
    mode: str = "agregado",
    normalizar: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Realiza una búsqueda aleatoria de hiperparámetros durante un número
    determinado de iteraciones.  Para cada configuración se ejecuta el
    pipeline completo y se almacena el macro F1‑score.  Al finalizar se
    devuelve un DataFrame ordenado por rendimiento y se actualiza la
    configuración óptima.

    Parameters
    ----------
    iterations : int
        Número de combinaciones aleatorias a evaluar.
    mode : str
        "agregado" o "replicas" para el preprocesamiento.
    normalizar : bool
        Si True aplica normalización min–max a las señales.
    verbose : bool
        Si True muestra información detallada por pantalla.

    Returns
    -------
    pandas.DataFrame
        Leaderboard con columnas [tipo_entropia, lF, d, tau, c, S_max, macro_f1]
        ordenado de mejor a peor.
    """
    # Definición de espacios de búsqueda
    tipo_entropia_options = [1, 2, 3, 4]
    lF_options = [1000, 1200, 1600]
    d_options = [3, 4]        # dimensión de embedding (o m)
    tau_options = [1, 2]
    c_options = [3, 4]        # número de símbolos para MDE/eMDE
    S_max_options = [10, 15]

    tried = set()
    results: List[Tuple[float, Tuple[int, int, int, int, int, int]]] = []
    for i in range(iterations):
        # Seleccionar aleatoriamente parámetros no probados previamente
        attempts = 0
        while True:
            candidate = (
                random.choice(tipo_entropia_options),
                random.choice(lF_options),
                random.choice(d_options),
                random.choice(tau_options),
                random.choice(c_options),
                random.choice(S_max_options),
            )
            if candidate not in tried:
                tried.add(candidate)
                break
            attempts += 1
            if attempts > 100:  # evitar bucle infinito si exhaustivo
                break
        tipo_entropia, lF, d, tau, c, S_max = candidate
        if verbose:
            print(
                f"\n[BUSQUEDA] Iteración {i+1}/{iterations} "
                f"→ Parámetros: tipo={tipo_entropia}, lF={lF}, d={d}, tau={tau}, c={c}, S_max={S_max}"
            )
        try:
            macro_f1 = evaluate_configuration(
                tipo_entropia, lF, d, tau, c, S_max, mode=mode, normalizar=normalizar
            )
        except Exception as e:
            # Registrar la configuración fallida con macro_f1=0
            macro_f1 = 0.0
            if verbose:
                print(f"[WARN] Configuración falló con error: {e}")
        results.append((macro_f1, candidate))
        if verbose:
            print(f"[RESULTADO] macro F1 = {macro_f1:.4f}\n")
    # Ordenar de mejor a peor
    results.sort(key=lambda x: x[0], reverse=True)
    # Construir DataFrame
    records = [
        {
            "tipo_entropia": cand[0],
            "lF": cand[1],
            "d": cand[2],
            "tau": cand[3],
            "c": cand[4],
            "S_max": cand[5],
            "macro_f1": f1,
        }
        for f1, cand in results
    ]
    leaderboard = pd.DataFrame(records)
    # Guardar leaderboard
    leaderboard_path = os.path.join("config", "param_leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    if verbose:
        print(f"[INFO] Leaderboard guardado en {leaderboard_path}")
    # Actualizar configuración óptima con la mejor entrada
    if results:
        best_f1, best_params = results[0]
        tipo_entropia, lF, d, tau, c, S_max = best_params
        # Guardar configuración óptima tanto en conf_temp como en conf_optimo
        ppr.guardar_configuracion(tipo_entropia, lF, d, tau, c, S_max)
        # Copiar a conf_optimo.csv para ser utilizada por ppr.ppr()
        config_values = [tipo_entropia, lF, d, tau, c, S_max]
        pd.DataFrame(config_values).to_csv(
            os.path.join("config", "conf_optimo.csv"),
            index=False,
            header=False,
        )
        if verbose:
            print(
                f"[MEJOR] tipo={tipo_entropia}, lF={lF}, d={d}, tau={tau}, c={c}, S_max={S_max}, "
                f"macro_f1={best_f1:.4f}"
            )
    return leaderboard


def run_optimum(verbose: bool = True, mode_preprocess: str = "agregado", normalizar: bool = True) -> None:
    """
    Ejecuta el pipeline completo utilizando la configuración óptima
    almacenada en ``config/conf_optimo.csv``.  A diferencia de la versión
    original que delegaba el preprocesamiento a ``ppr.ppr()``, esta
    implementación aprovecha la función ``evaluate_configuration`` para
    recalcular las características y etiquetas conforme a los parámetros
    óptimos y al modo de preprocesamiento deseado.  Esto asegura que
    siempre se usen todas las réplicas y que la segmentación siga la
    lógica de vibraciones de motor.

    Parameters
    ----------
    verbose : bool
        Si True muestra información detallada.
    mode_preprocess : str
        "agregado" o "replicas".  Determina si en la ejecución óptima se
        promedian las entropías de las réplicas (``agregado``) o se
        tratan de forma independiente (``replicas``).
    normalizar : bool
        Si True aplica normalización min–max a las señales antes de la
        derivada.
    """
    # Leer configuración óptima
    conf_path = os.path.join("config", "conf_optimo.csv")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(
            f"No se encontró {conf_path}. Ejecute primero el modo de búsqueda para generar una configuración óptima."
        )
    conf_values = pd.read_csv(conf_path, header=None).values.flatten().astype(int)
    tipo_entropia, lF, d, tau, c, S_max = conf_values
    if verbose:
        print(
            f"\n[RUN] Ejecutando pipeline con configuración óptima: "
            f"tipo={tipo_entropia}, lF={lF}, d={d}, tau={tau}, c={c}, S_max={S_max}\n"
        )
    # Ejecutar evaluación completa con la configuración óptima
    macro_f1 = evaluate_configuration(
        tipo_entropia,
        lF,
        d,
        tau,
        c,
        S_max,
        mode=mode_preprocess,
        normalizar=normalizar,
    )
    if verbose:
        print(f"\n[RUN] Macro F1-score obtenido con configuración óptima: {macro_f1:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros y ejecución con configuración óptima.")
    parser.add_argument(
        "--mode",
        choices=["search", "run"],
        default="run",
        help="Modo de ejecución: 'search' para buscar parámetros, 'run' para ejecutar con la configuración óptima",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Número de combinaciones aleatorias a evaluar en modo 'search' (por defecto 5)",
    )
    parser.add_argument(
        "--mode-preprocess",
        choices=["agregado", "replicas"],
        default="agregado",
        help="Modo de preprocesamiento para búsqueda de parámetros",
    )
    parser.add_argument(
        "--no-normalizar",
        action="store_true",
        help="Desactiva la normalización min–max durante la búsqueda",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Ejecuta sin imprimir información detallada",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    if args.mode == "search":
        search_params(
            iterations=args.iterations,
            mode=args.mode_preprocess,
            normalizar=(not args.no_normalizar),
            verbose=verbose,
        )
    else:
        run_optimum(
            verbose=verbose,
            mode_preprocess=args.mode_preprocess,
            normalizar=(not args.no_normalizar),
        )


if __name__ == "__main__":
    main()
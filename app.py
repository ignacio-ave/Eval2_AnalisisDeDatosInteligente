"""
main_simple.py
Ejecución modular y comparativa de entropías multi-escala.

Modos disponibles:
  1 → Prueba manual de parámetros.
  2 → Reejecutar la mejor configuración guardada (del leaderboard).
  3 → Comparar las 4 entropías con la misma configuración.

Cada ejecución:
  - Crea carpetas por tipo de entropía (requerimiento).
  - Calcula F1 y guarda resultados.
  - Actualiza leaderboard global y mejor configuración.
"""

import os
import csv
from datetime import datetime
import pandas as pd
from main import evaluate_configuration

# =======================================
# CONFIGURACIÓN Y CARPETAS
# =======================================
CARPETAS = {
    1: "Entropia_MDE",
    2: "Entropia_MDE_Mejorada",
    3: "Entropia_MPE",
    4: "Entropia_MPE_Mejorada"
}

LEADERBOARD_PATH = "leaderboard/leaderboard.csv"
BEST_CONFIG_PATH = "leaderboard/mejor_config.csv"


# =======================================
# FUNCIONES DE APOYO
# =======================================
def crear_carpetas_resultados():
    for _, nombre in CARPETAS.items():
        os.makedirs(nombre, exist_ok=True)
    os.makedirs("leaderboard", exist_ok=True)
    print("[INFO] Carpetas creadas correctamente.")


def registrar_leaderboard(config, f1_score):
    """Registra la configuración y su F1-score en el leaderboard global."""
    nueva_fila = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo_entropia": config["tipo_entropia"],
        "lF": config["lF"],
        "d": config["d"],
        "tau": config["tau"],
        "c": config["c"],
        "S_max": config["S_max"],
        "F1": f1_score
    }

    if os.path.exists(LEADERBOARD_PATH):
        df = pd.read_csv(LEADERBOARD_PATH)
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
    else:
        df = pd.DataFrame([nueva_fila])

    df = df.sort_values(by="F1", ascending=False)
    df.to_csv(LEADERBOARD_PATH, index=False)

    # Guardar mejor configuración
    mejor = df.iloc[0]
    mejor.to_frame().T.to_csv(BEST_CONFIG_PATH, index=False)
    print(f"[INFO] Leaderboard actualizado ({len(df)} registros totales).")
    print(f"[INFO] Mejor F1 actual: {mejor['F1']:.4f}")


def leer_mejor_config():
    """Lee la mejor configuración del leaderboard."""
    if not os.path.exists(BEST_CONFIG_PATH):
        print("[WARN] No hay configuración previa.")
        return None

    df = pd.read_csv(BEST_CONFIG_PATH)
    fila = df.iloc[0]
    return {
        "tipo_entropia": int(fila["tipo_entropia"]),
        "lF": int(fila["lF"]),
        "d": int(fila["d"]),
        "tau": int(fila["tau"]),
        "c": int(fila["c"]),
        "S_max": int(fila["S_max"]),
        "mode": "agregado",
        "normalizar": True
    }


# =======================================
# MODO 1 – EJECUCIÓN MANUAL
# =======================================
def modo_manual():
    try:
        tipo_entropia = int(input("Tipo de entropía (1–4, por defecto 1): ") or 1)
        lF = int(input("Longitud de segmento (1000/1200/1600, por defecto 1000): ") or 1000)
        d = int(input("Dimensión embebida d (por defecto 3): ") or 3)
        tau = int(input("Retardo τ (por defecto 1): ") or 1)
        c = int(input("Símbolos c (por defecto 3): ") or 3)
        S_max = int(input("Máximo de escalas (por defecto 10): ") or 10)
    except ValueError:
        print("[ERROR] Parámetros inválidos. Cancelando ejecución.")
        return

    config = {
        "tipo_entropia": tipo_entropia,
        "lF": lF,
        "d": d,
        "tau": tau,
        "c": c,
        "S_max": S_max,
        "mode": "agregado",
        "normalizar": True
    }

    print("\n[INFO] Ejecutando configuración:")
    print(config)

    f1_score = evaluate_configuration(
        config["tipo_entropia"], config["lF"], config["d"],
        config["tau"], config["c"], config["S_max"],
        mode=config["mode"], normalizar=config["normalizar"]
    )

    print(f"\n✅ F1 obtenido: {f1_score:.4f}")
    registrar_leaderboard(config, f1_score)

    carpeta = CARPETAS.get(tipo_entropia, "Resultados")
    archivo_f1 = os.path.join(carpeta, f"resultado_F1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(archivo_f1, "w") as f:
        f.write(f"Configuración: {config}\nF1: {f1_score:.4f}\n")
    print(f"[INFO] Resultados guardados en: {archivo_f1}")


# =======================================
# MODO 2 – MEJOR CONFIGURACIÓN GUARDADA
# =======================================
def modo_mejor():
    config = leer_mejor_config()
    if not config:
        return

    print(f"\n[INFO] Ejecutando mejor configuración:\n{config}")
    f1_score = evaluate_configuration(
        config["tipo_entropia"], config["lF"], config["d"],
        config["tau"], config["c"], config["S_max"],
        mode=config["mode"], normalizar=config["normalizar"]
    )
    print(f"\n✅ F1 obtenido (mejor config): {f1_score:.4f}")


# =======================================
# MODO 3 – COMPARAR LAS 4 ENTROPÍAS
# =======================================
def modo_comparar():
    print("\n[INFO] Comparación de las 4 entropías con mismos parámetros.")
    try:
        lF = int(input("Longitud de segmento (1000/1200/1600, por defecto 1000): ") or 1000)
        d = int(input("Dimensión embebida d (por defecto 3): ") or 3)
        tau = int(input("Retardo τ (por defecto 1): ") or 1)
        c = int(input("Símbolos c (por defecto 3): ") or 3)
        S_max = int(input("Máximo de escalas (por defecto 10): ") or 10)
    except ValueError:
        print("[ERROR] Parámetros inválidos.")
        return

    resultados = []

    for tipo_entropia in range(1, 5):
        config = {
            "tipo_entropia": tipo_entropia,
            "lF": lF,
            "d": d,
            "tau": tau,
            "c": c,
            "S_max": S_max,
            "mode": "agregado",
            "normalizar": True
        }

        print(f"\n[INFO] Ejecutando tipo_entropia = {tipo_entropia}")
        f1_score = evaluate_configuration(
            config["tipo_entropia"], config["lF"], config["d"],
            config["tau"], config["c"], config["S_max"],
            mode=config["mode"], normalizar=config["normalizar"]
        )
        resultados.append((tipo_entropia, f1_score))
        registrar_leaderboard(config, f1_score)

    print("\n=== RESULTADOS COMPARATIVOS ===")
    for tipo, f1 in resultados:
        print(f"Entropía {tipo}: F1 = {f1:.4f}")

    mejor_tipo, mejor_f1 = max(resultados, key=lambda x: x[1])
    print(f"\n🏆 Mejor entropía: {mejor_tipo} con F1 = {mejor_f1:.4f}")


# =======================================
# MAIN
# =======================================
def main():
    crear_carpetas_resultados()

    print("\n=== MENÚ DE EJECUCIÓN ===")
    print("1 → Ejecutar configuración manual (nueva prueba)")
    print("2 → Ejecutar mejor configuración actual")
    print("3 → Comparar las 4 entropías con parámetros fijos")

    opcion = input("Seleccione modo [1/2/3]: ").strip()
    if opcion == "1":
        modo_manual()
    elif opcion == "2":
        modo_mejor()
    elif opcion == "3":
        modo_comparar()
    else:
        print("[ERROR] Opción inválida.")


if __name__ == "__main__":
    main()

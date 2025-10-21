"""
main_simple.py
Ejecuta el flujo completo con los parámetros óptimos encontrados.

Modo:
    python main_simple.py
"""

import os
from main import evaluate_configuration

def main():
    # --- Parámetros óptimos ---
    tipo_entropia = 1   # 1 = MDE
    lF = 1000           # longitud de ventana
    d = 3               # embedding dimension
    tau = 1             # delay
    c = 3               # símbolos
    S_max = 10          # escalas
    mode_preprocess = "agregado"
    normalizar = True

    print("\n[MAIN SIMPLE] Ejecutando configuración óptima...")
    print(f"tipo_entropia={tipo_entropia}, lF={lF}, d={d}, tau={tau}, c={c}, S_max={S_max}")
    print(f"Preprocesamiento: {mode_preprocess}, Normalizar: {normalizar}\n")

    # --- Ejecutar pipeline completo ---
    f1_score = evaluate_configuration(
        tipo_entropia, lF, d, tau, c, S_max,
        mode=mode_preprocess,
        normalizar=normalizar
    )

    # --- Guardar resultados en config/conf_optimo.csv ---
    os.makedirs("config", exist_ok=True)
    with open("config/conf_optimo.csv", "w") as f:
        f.write(f"{tipo_entropia}\n{lF}\n{d}\n{tau}\n{c}\n{S_max}\n")

    print("\n✅ Ejecución completada")
    print(f"🔹 F1 obtenido: {f1_score:.4f}")
    print("Configuración guardada en config/conf_optimo.csv\n")

if __name__ == "__main__":
    main()

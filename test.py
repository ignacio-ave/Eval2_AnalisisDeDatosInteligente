#
#      .                          .                              
#    .o8                        .o8                              
#  .o888oo  .ooooo.   .oooo.o .o888oo     oo.ooooo.  oooo    ooo 
#    888   d88' `88b d88(  "8   888        888' `88b  `88.  .8'  
#    888   888ooo888 `"Y88b.    888        888   888   `88..8'   
#    888 . 888    .o o.  )88b   888 . .o.  888   888    `888'    
#    "888" `Y8bod8P' 8""888P'   "888" Y8P  888bod8P'     .8'     
#                                          888       .o..P'      
#                                         o888o      `Y8P'       
#                                                              
#


import numpy as np
import os
import pandas as pd

DATA_DIR = "data"

def cargar_datos_tst(ruta_X_test, ruta_y_test, ruta_W):
    print("[ETAPA] Cargando datos de prueba y pesos...")
    X_test = pd.read_csv(ruta_X_test, header=None).values.astype(float)
    y_test = pd.read_csv(ruta_y_test, header=None).values.astype(float)
    W = pd.read_csv(ruta_W, header=None).values.astype(float)

    print(f"[INFO] X_test shape = {X_test.shape}")
    print(f"[INFO] y_test shape = {y_test.shape}")
    print(f"[INFO] W shape = {W.shape}")

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test y y_test deben tener el mismo número de filas")
    return X_test, y_test, W

def predecir_softmax(X_test, W):
    print("[ETAPA] Calculando probabilidades con softmax...")
    Xb = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    z = Xb @ W
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    y_pred_proba = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return y_pred_proba

def obtener_etiquetas_predichas(y_pred_proba):
    return np.argmax(y_pred_proba, axis=1)

def matriz_confusion(y_true, y_pred):
    print("[ETAPA] Calculando matriz de confusión...")
    cm = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            cm[i, j] = np.sum((y_true == i) & (y_pred == j))
    print("[INFO] Matriz de confusión:\n", cm)
    return cm

def calcular_fscores(cm):
    print("[ETAPA] Calculando F-scores por clase...")
    fscores = np.zeros(4)
    for i in range(4):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fscores[i] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"[INFO] Clase {i}: Precision = {precision:.4f}, Recall = {recall:.4f}, F-score = {fscores[i]:.4f}")
    print(f"[INFO] Macro F-score promedio = {np.mean(fscores):.4f}")
    return fscores

def guardar_resultados(cm, fscores, ruta_cm, ruta_fs):
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(cm).to_csv(ruta_cm, index=False, header=False)
    pd.DataFrame(fscores).to_csv(ruta_fs, index=False, header=False)
    print(f"[ok] Resultados guardados en:\n - {ruta_cm}\n - {ruta_fs}")

def test():
    print("\n========================")
    print("   EVALUACIÓN DEL MODELO")
    print("========================\n")

    # Cargar datos
    X_test, y_test, W = cargar_datos_tst(
        os.path.join(DATA_DIR, "dtst.csv"),
        os.path.join(DATA_DIR, "dtst_label.csv"),
        os.path.join(DATA_DIR, "pesos.csv")
    )

    # Convertir etiquetas one-hot a clases
    y_test_labels = np.argmax(y_test, axis=1)

    # Predicción
    y_pred_proba = predecir_softmax(X_test, W)
    y_pred = obtener_etiquetas_predichas(y_pred_proba)

    # Matriz de confusión y F-scores
    cm = matriz_confusion(y_test_labels, y_pred)
    fscores = calcular_fscores(cm)

    # Guardar resultados
    guardar_resultados(
        cm, fscores,
        os.path.join(DATA_DIR, "cmatriz.csv"),
        os.path.join(DATA_DIR, "fscores.csv")
    )

    print("\n[ok] Evaluación finalizada.\n")

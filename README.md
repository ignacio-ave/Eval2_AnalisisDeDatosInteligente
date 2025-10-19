# TAREA_1310: Clasificación Multi-Clases con Entropía Multi-Escala y Reg. Softmax

**Profesor:** Nibaldo Rodríguez A.
**Fecha de Entrega:** Viernes 24 de Octubre del 2025, 09:00 pm
**Lugar:** Aula Virtual del curso
**Requerimientos Técnicos:**
- Python 3.12 (Anaconda)
- Librerías: pandas, numpy

---

## Objetivos

| Tipo | Descripción |
|------|-------------|
| **General** | Implementar y evaluar el rendimiento de un modelo de clasificación multi-clases usando entropía multi-escalas y regresión softmax |
| **Específico 1** | Implementar el proceso de entrenamiento y test usando algoritmo descenso del gradiente con momentum |
| **Específico 2** | Evaluar rendimiento mediante Matriz de Confusión y F-scores por clase |

---

## Estructura de Archivos

| Archivo | Descripción |
|---------|-------------|
| `ppr.py` | Pre-procesamiento de datos |
| `trn.py` | Algoritmo de entrenamiento |
| `tst.py` | Evaluación de rendimiento |
| `conf_ppr.csv` | Configuración de pre-procesamiento |
| `conf_train.csv` | Configuración de entrenamiento |

---

## Requerimientos Funcionales y No Funcionales

### 1. Pre-Procesamiento de Datos (`ppr.py`)

| RF | Descripción | Entradas | Salidas | Parámetros | Requerimientos No Funcionales | Ejemplo |
|----|-------------|----------|---------|------------|-------------------------------|---------|
| RF-001 | Cargar datos de clases | Rutas a Class1-4.csv | Matriz X ∈ ℝ^(120000,4) | rutas (lista) | RNF-001: 120000 filas/archivo <br> RNF-002: Sin NaN | X = cargar_datos(["Class1.csv", ...]) |
| RF-002 | Aplicar diferencia finita | Matriz X ∈ ℝ^(120000,4) | Matriz X_diff ∈ ℝ^(119999,4) | X (matriz) | RNF-003: Usar np.diff <br> RNF-004: Validar 119999 filas | X_diff = aplicar_diferencia_finita(X) |
| RF-003 | Segmentar muestras | X_diff, lF ∈ {1000,1200,1600} | Lista de segmentos | X_diff, lF | RNF-005: Divisible por lF <br> RNF-006: Sin solapamiento | segmentos = segmentar_muestras(X_diff, 1000) |
| RF-004 | Calcular entropías | Segmentos, d, tau, c, Smax | dClases.csv | segmentos, d, tau, c, Smax | RNF-007: Fórmulas según slides <br> RNF-008: Parámetros enteros | dClases = calcular_entropias(segmentos, 2,1,3,10) |
| RF-005 | Generar etiquetas | Nº muestras/clase | dLabel.csv | n_muestras | RNF-009: Coincidencia filas-segmentos | dLabel = generar_etiquetas(30000) |
| RF-006 | Guardar configuración | Parámetros | conf_ppr.csv | tipo, lF, d, tau, c, Smax | RNF-010: Parámetros positivos | guardar_configuracion(1,1000,2,1,3,10) |

### 2. Algoritmo de Entrenamiento (`trn.py`)

| RF | Descripción | Entradas | Salidas | Parámetros | Requerimientos No Funcionales | Ejemplo |
|----|-------------|----------|---------|------------|-------------------------------|---------|
| RF-001 | Cargar datos | dClases.csv, dLabel.csv | X, y | rutas | RNF-001: Mismo Nº filas | X, y = cargar_datos("dClases.csv", "dLabel.csv") |
| RF-002 | Reordenar aleatoriamente | X, y | X_shuffled, y_shuffled | seed=42 | RNF-002: Orden sincronizado | X_shuf, y_shuf = reordenar_aleatoriamente(X, y) |
| RF-003 | Normalizar Z-score | X | X_norm | - | RNF-003: Sin división por cero | X_norm = normalizar_zscore(X) |
| RF-004 | Dividir train/test | X_norm, y_shuffled, p | X_train, X_test, y_train, y_test | p (0.65-0.80) | RNF-004: Validar rango p | X_train, X_test, y_train, y_test = dividir_train_test(X_norm, y_shuffled, 0.75) |
| RF-005 | Entrenar mGD | X_train, y_train, max_iter, mu, momentum | W, J | max_iter, mu, momentum | RNF-005: Costos decrecientes | W, J = entrenar_mgd(X_train, y_train, 1000, 0.01, 0.9) |
| RF-006 | Guardar pesos y costo | W, J | pesos.csv, costo.csv | rutas | RNF-006: Validar dimensiones | guardar_pesos_y_costo(W, J, "pesos.csv", "costo.csv") |
| RF-007 | Cargar configuración | conf_train.csv | max_iter, mu, p | ruta | RNF-007: Valores numéricos | max_iter, mu, p = cargar_configuracion("conf_train.csv") |

### 3. Evaluación de Rendimiento (`tst.py`)

| RF | Descripción | Entradas | Salidas | Parámetros | Requerimientos No Funcionales | Ejemplo |
|----|-------------|----------|---------|------------|-------------------------------|---------|
| RF-001 | Cargar datos test | dtst.csv, dtst_label.csv, pesos.csv | X_test, y_test, W | rutas | RNF-001: Dimensiones compatibles | X_test, y_test, W = cargar_datos("dtst.csv", "dtst_label.csv", "pesos.csv") |
| RF-002 | Predecir softmax | X_test, W | y_pred_proba | - | RNF-002: Implementar softmax <br> RNF-003: Sin overflow | y_pred_proba = predecir_softmax(X_test, W) |
| RF-003 | Obtener etiquetas | y_pred_proba | y_pred | - | RNF-004: Usar argmax | y_pred = obtener_etiquetas_predichas(y_pred_proba) |
| RF-004 | Matriz de confusión | y_test, y_pred | cm (4x4) | - | RNF-005: Inicializar ceros | cm = matriz_confusion(y_test, y_pred) |
| RF-005 | Calcular F-scores | cm | fscores (1x4) | - | RNF-006: Calcular precision/recall | fscores = calcular_fscores(cm) |
| RF-006 | Guardar resultados | cm, fscores | cmatriz.csv, fscores.csv | rutas | RNF-007: Validar dimensiones | guardar_resultados(cm, fscores, "cmatriz.csv", "fscores.csv") |

---

## Configuración de Archivos

### conf_ppr.csv
| Línea | Contenido |
|-------|-----------|
| 1 | Tipo de entropía (1-4) |
| 2 | Longitud del segmento (lF) |
| 3 | Dimensión embebida (d) |
| 4 | Factor de retardo (tau) |
| 5 | Número de símbolos (c) |
| 6 | Número máximo de escalas (Smax) |

### conf_train.csv
| Línea | Contenido |
|-------|-----------|
| 1 | Máximo de iteraciones |
| 2 | Tasa de aprendizaje (mu) |
| 3 | Porcentaje de training (65-80%) |

---

## Observaciones
Si un grupo no cumple los requerimientos funcionales y no-funcionales, la escala de evaluación será entre 1.0 y 3.0.

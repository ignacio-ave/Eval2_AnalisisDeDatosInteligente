TAREA_1310: Clasificación Multi-Clases con Entropía Multi-Escala y Reg. Softmax

OBJETIVOS
General:
Implementar y evaluar el rendimiento de un modelo de clasificación multi-clases usando entropía multi-escalas y regresión softmax.
Específicos:
1. Implementar el proceso de entrenamiento y test usando algoritmo descenso del gradiente con momentum.
2. Evaluar rendimiento del modelo mediante Matriz de Confusión y F-scores para cada clase.

REQUERIMIENTOS FUNCIONALES Y NO FUNCIONALES

1. PRE-PROCESAMIENTO DE DATOS (ppr.py)

RF-001: cargar_datos
Descripción: Carga los archivos Class1.csv a Class4.csv (4-muestras de N=120000 valores). Notación: X ∈ ℝ^(N,4), N=120000.
Entradas: Rutas a Class1.csv, Class2.csv, Class3.csv, Class4.csv.
Salidas: Matriz X ∈ ℝ^(120000,4).
Requerimientos No Funcionales:
RNF-001: Validar que cada archivo tenga exactamente 120000 filas y 1 columna.
RNF-002: Validar ausencia de valores NaN.
Ejemplo: X = cargar_datos(["Class1.csv", "Class2.csv", "Class3.csv", "Class4.csv"])

RF-002: aplicar_diferencia_finita
Descripción: Aplica diferencias finitas: x(n) = x(n) - x(n-1), n=2,...,N.
Entradas: Matriz X ∈ ℝ^(120000,4).
Salidas: Matriz X_diff ∈ ℝ^(119999,4).
Requerimientos No Funcionales:
RNF-003: Usar np.diff(X, axis=0).
RNF-004: Validar salida con 119999 filas.
Ejemplo: X_diff = aplicar_diferencia_finita(X)

RF-003: segmentar_muestras
Descripción: Segmenta muestras con longitud lF ∈ {1000, 1200, 1600}. Notación: x_i ∈ ℝ^(lF, nF), nF = N/lF.
Entradas: Matriz X_diff ∈ ℝ^(119999,4), lF (longitud del segmento).
Salidas: Lista de matrices segmentadas.
Requerimientos No Funcionales:
RNF-005: Validar que 119999 sea divisible por lF.
RNF-006: Segmentar por filas, sin solapamiento.
Ejemplo: segmentos = segmentar_muestras(X_diff, lF=1000)

RF-004: calcular_entropias
Descripción: Calcula entropías multi-escala (MDE, eMDE, MPE, eMPE) por segmento.
Entradas: Lista de segmentos, parámetros d, tau, c, Smax.
Salidas: DataFrame de entropías concatenadas (dClases.csv).
Requerimientos No Funcionales:
RNF-007: Implementar fórmulas según slides del profesor.
RNF-008: Validar parámetros enteros positivos.
Ejemplo: dClases = calcular_entropias(segmentos, d=2, tau=1, c=3, Smax=10)

RF-005: generar_etiquetas
Descripción: Crea etiquetas binarias para 4 clases: (1 0 0 0), (0 1 0 0), etc.
Entradas: Número de muestras por clase.
Salidas: DataFrame de etiquetas (dLabel.csv).
Requerimientos No Funcionales:
RNF-009: Validar coincidencia de filas con segmentos.
Ejemplo: dLabel = generar_etiquetas(n_muestras_por_clase=30000)

RF-006: guardar_configuracion
Descripción: Guarda configuración en conf_ppr.csv (6 líneas).
Entradas: tipo_entropia, lF, d, tau, c, Smax.
Salidas: Archivo conf_ppr.csv.
Requerimientos No Funcionales:
RNF-010: Validar parámetros enteros positivos.
Ejemplo: guardar_configuracion(1, 1000, 2, 1, 3, 10)

2. ALGORITMO DE ENTRENAMIENTO (trn.py)

RF-001: cargar_datos
Descripción: Carga dClases.csv y dLabel.csv.
Entradas: Rutas a dClases.csv y dLabel.csv.
Salidas: Matrices X (características) y y (etiquetas).
Requerimientos No Funcionales:
RNF-001: Validar mismo número de filas en X y y.
Ejemplo: X, y = cargar_datos("dClases.csv", "dLabel.csv")

RF-002: reordenar_aleatoriamente
Descripción: Reordena aleatoriamente X y y sincronizadamente.
Entradas: Matrices X y y.
Salidas: Matrices X_shuffled y y_shuffled.
Requerimientos No Funcionales:
RNF-002: Usar np.random.seed(42).
RNF-003: Validar orden sincronizado.
Ejemplo: X_shuffled, y_shuffled = reordenar_aleatoriamente(X, y)

RF-003: normalizar_zscore
Descripción: Normaliza características: x = (x - mean(x)) / std(x).
Entradas: Matriz X.
Salidas: Matriz X_norm normalizada.
Requerimientos No Funcionales:
RNF-004: Validar ausencia de divisiones por cero.
Ejemplo: X_norm = normalizar_zscore(X)

RF-004: dividir_train_test
Descripción: Divide datos en entrenamiento (65-80%) y prueba.
Entradas: X_norm, y_shuffled, porcentaje p (65 < p < 81).
Salidas: X_train, X_test, y_train, y_test.
Requerimientos No Funcionales:
RNF-005: Validar p ∈ [0.65, 0.80].
Ejemplo: X_train, X_test, y_train, y_test = dividir_train_test(X_norm, y_shuffled, 0.75)

RF-005: entrenar_mgd
Descripción: Entrena modelo con mGD (descenso de gradiente con momentum).
Entradas: X_train, y_train, max_iter, mu, momentum.
Salidas: Matriz de pesos W, vector de costo J.
Requerimientos No Funcionales:
RNF-006: Inicializar W aleatoriamente.
RNF-007: Validar disminución monótona del costo.
Ejemplo: W, J = entrenar_mgd(X_train, y_train, 1000, 0.01, 0.9)

RF-006: guardar_pesos_y_costo
Descripción: Guarda pesos y costos en pesos.csv y costo.csv.
Entradas: W, J, rutas de salida.
Salidas: Archivos pesos.csv y costo.csv.
Requerimientos No Funcionales:
RNF-008: Validar dimensiones de W y J.
Ejemplo: guardar_pesos_y_costo(W, J, "pesos.csv", "costo.csv")

RF-007: cargar_configuracion
Descripción: Carga configuración desde conf_train.csv.
Entradas: Ruta a conf_train.csv.
Salidas: max_iter, mu, p.
Requerimientos No Funcionales:
RNF-009: Validar valores numéricos positivos.
Ejemplo: max_iter, mu, p = cargar_configuracion("conf_train.csv")

3. EVALUACIÓN DE RENDIMIENTO (tst.py)

RF-001: cargar_datos
Descripción: Carga dtst.csv, dtst_label.csv y pesos.csv.
Entradas: Rutas a los archivos.
Salidas: X_test, y_test, W.
Requerimientos No Funcionales:
RNF-001: Validar compatibilidad de dimensiones.
Ejemplo: X_test, y_test, W = cargar_datos("dtst.csv", "dtst_label.csv", "pesos.csv")

RF-002: predecir_softmax
Descripción: Calcula probabilidades con regresión softmax.
Entradas: X_test, W.
Salidas: Matriz de probabilidades y_pred_proba.
Requerimientos No Funcionales:
RNF-002: Implementar softmax manualmente.
RNF-003: Validar ausencia de overflow.
Ejemplo: y_pred_proba = predecir_softmax(X_test, W)

RF-003: obtener_etiquetas_predichas
Descripción: Obtiene etiquetas predichas desde probabilidades.
Entradas: y_pred_proba.
Salidas: Vector de etiquetas y_pred.
Requerimientos No Funcionales:
RNF-004: Usar np.argmax.
Ejemplo: y_pred = obtener_etiquetas_predichas(y_pred_proba)

RF-004: matriz_confusion
Descripción: Calcula matriz de confusión (4x4).
Entradas: y_test, y_pred.
Salidas: Matriz de confusión cm ∈ ℝ^(4,4).
Requerimientos No Funcionales:
RNF-005: Inicializar matriz de ceros.
Ejemplo: cm = matriz_confusion(y_test, y_pred)

RF-005: calcular_fscores
Descripción: Calcula F-scores por clase: F1 = 2 × (precision × recall) / (precision + recall).
Entradas: Matriz de confusión cm.
Salidas: Vector de F-scores fscores ∈ ℝ^(1,4).
Requerimientos No Funcionales:
RNF-006: Calcular precision y recall por clase.
Ejemplo: fscores = calcular_fscores(cm)

RF-006: guardar_resultados
Descripción: Guarda resultados en cmatriz.csv y fscores.csv.
Entradas: cm, fscores, rutas de salida.
Salidas: Archivos cmatriz.csv y fscores.csv.
Requerimientos No Funcionales:
RNF-007: Validar dimensiones de cm y fscores.
Ejemplo: guardar_resultados(cm, fscores, "cmatriz.csv", "fscores.csv")

ENTREGA
Fecha: Viernes 24 de Octubre del 2025
Hora: 09:00 pm.
Lugar: Aula Virtual del curso.
Requerimientos Técnicos:
- Python 3.12 (Anaconda)
- Librerías: pandas, numpy

OBSERVACIÓN
Si un grupo no cumple los requerimientos funcionales y no-funcionales, la escala de evaluación será entre 1.0 y 3.0.

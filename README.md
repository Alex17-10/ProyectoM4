# PROYECTO INTEGRADOR — CHURN BANCARIO

## AVANCE 4 — DOCUMENTACIÓN Y CONSOLIDACIÓN

## Descripción del proyecto

FinanceGuard es un banco digital que ha experimentado un crecimiento exponencial en los últimos años, alcanzando 10000 clientes activos. Sin embargo, la dirección ha detectado un incremento preocupante en la tasa de abandono de clientes (churn), que actualmente alcanza el 20% anual, resultando en pérdidas millonarias.

Primero se comprendió la estructura y calidad de los datos; después se construyeron y compararon modelos de clasificación para estimar el churn; finalmente se aplicaron técnicas de aprendizaje no supervisado para segmentar clientes y estudiar cómo los segmentos se relacionan con la variable objetivo.

El Avance 4 no incorpora un nuevo modelo ni pretende demostrar un análisis adicional. Su función es consolidar lo que ya fue ejecutado, explicar las decisiones tomadas, presentar los resultados relevantes y dejar una ruta clara para reproducir o continuar el proyecto. La documentación se limita a procedimientos y resultados respaldados por los scripts, el dataset y los resultados disponibles.

## Problema abordado

La variable Exited representa si un cliente abandonó la entidad. El problema combina dos necesidades complementarias: identificar qué tan bien pueden distinguirse clientes con y sin churn mediante modelos supervisados, y descubrir grupos de clientes con características similares mediante clustering. La segmentación no sustituye la predicción; funciona como una perspectiva adicional para interpretar la heterogeneidad de la base.

## Objetivo

Como Científico de Datos Junior, tu misión es desarrollar un modelo de Machine Learning completo que prediga qué clientes tienen mayor probabilidad de abandonar el banco. Este modelo permitirá al equipo de retención implementar estrategias personalizadas para retener a los clientes en riesgo, con el objetivo de reducir el churn al 15%.

## Alcance del Avance 4

La documentación reúne el flujo completo desde el dataset de entrada hasta el dataset segmentado. Incluye preparación de datos, división entrenamiento/prueba, validación cruzada, modelos supervisados, optimización de XGBoost, interpretabilidad de la regresión logística, comparación de modelos, K-Means, DBSCAN, PCA, t-SNE, relación entre segmentos y churn, exportación de resultados y reproducibilidad.

## Datos y preparación

El archivo de entrada utilizado fue Churn_Modelling.csv. Contiene 10.000 registros y 14 variables. La validación realizada en el Avance 3 confirmó que las variables analizadas no presentan valores nulos y que no existen filas duplicadas. La misma base fue utilizada como referencia para los procesos de clasificación y clustering.

### Variables del dataset

| Variable | Tipo | Uso principal |
|---|---|---|
| RowNumber | Numérica | Identificador técnico; excluida del modelado |
| CustomerId | Numérica | Identificador del cliente; excluida del modelado |
| Surname | Categórica | Dato nominal; excluido del modelado |
| CreditScore | Numérica | Predictora / clustering |
| Geography | Categórica | Predictora / clustering |
| Gender | Categórica | Predictora / clustering |
| Age | Numérica | Predictora / clustering |
| Tenure | Numérica | Predictora / clustering |
| Balance | Numérica | Predictora / clustering |
| NumOfProducts | Numérica | Predictora / clustering |
| HasCrCard | Numérica binaria | Predictora / clustering |
| IsActiveMember | Numérica binaria | Predictora / clustering |
| EstimatedSalary | Numérica | Predictora / clustering |
| Exited | Binaria | Variable objetivo; no usada para formar clusters |

En los modelos supervisados se eliminaron RowNumber, CustomerId y Surname y se utilizó Exited como variable objetivo. Las variables numéricas se estandarizaron mediante StandardScaler y las categóricas mediante OneHotEncoder con eliminación de la primera categoría. En el clustering se mantuvieron las variables binarias en escala 0–1 antes de la estandarización final y también se aplicó OneHotEncoder a Geography y Gender.

### Distribución inicial del churn

> **[Imagen del PDF: gráfico de barras "Distribución de clientes según Churn" (eje X: Estado del cliente — "No abandonó" / "Abandonó"; eje Y: Cantidad de clientes).]**

Distribución de la variable objetivo en los 10.000 clientes.

La base contiene 7.963 clientes sin churn y 2.037 con churn, equivalentes a 79,63% y 20,37%, respectivamente. Esta distribución confirma que la clase churn es minoritaria, por lo que la evaluación no debe depender únicamente de Accuracy.

## Continuidad de los tres avances

El proyecto debe entenderse como un único flujo analítico. Cada avance respondió a una etapa diferente del problema y los resultados de una etapa sirven de base para la siguiente.

### Comprensión y baseline

El primer avance se concentró en el análisis exploratorio, la calidad del dataset, la clasificación de variables, la preparación de los datos y la construcción de una Regresión Logística como modelo baseline. También se estudiaron correlaciones, multicolinealidad mediante VIF, coeficientes, Odds Ratio e intervalos de confianza.

### Mejora de la predicción

El segundo avance amplió el modelado con Random Forest, XGBoost, XGBoost optimizado, LightGBM y CatBoost. Se incorporó validación cruzada estratificada de cinco folds, Grid Search para XGBoost, comparación de métricas y un modelo de Stacking formado por XGBoost, LightGBM y CatBoost con Regresión Logística como meta-learner.

### Segmentación de clientes

El tercer avance trasladó el análisis al aprendizaje no supervisado. Se evaluó el número de clusters para K-Means mediante Inertia y Silhouette, se construyeron tres segmentos, se perfiló cada grupo y se compararon sus tasas de churn. Después se evaluó DBSCAN con distintas combinaciones de eps y min_samples. PCA y t-SNE se utilizaron como herramientas de reducción de dimensionalidad y visualización.

Esta secuencia permite pasar de una pregunta general —qué caracteriza el churn— a dos perspectivas complementarias: quién puede ser clasificado como cliente de mayor riesgo y cómo se agrupan los clientes según sus características.

## Análisis exploratorio y modelo baseline

El análisis inicial estableció una base cuantitativa para comprender el problema. Se revisaron dimensiones, valores faltantes, duplicados, estadísticas descriptivas y variables categóricas. La base no presentó valores faltantes en las variables utilizadas en el análisis de clustering ni duplicados.

La preparación para clasificación separó las variables predictoras de Exited y descartó identificadores que no aportaban una característica conductual o financiera útil para el modelo. La división train/test se realizó con 80% de los datos para entrenamiento y 20% para prueba, usando random_state=42 y estratificación para conservar la proporción de churn.

| Conjunto | Registros | Proporción | Churn | No churn |
|---|---|---|---|---|
| Entrenamiento | 8.000 | 80% | 1.630 | 6.370 |
| Prueba | 2.000 | 20% | 407 | 1.593 |
| Total | 10.000 | 100% | 2.037 | 7.963 |

### Regresión Logística baseline

| Métrica | Resultado |
|---|---|
| Accuracy | 0,8080 |
| Precision | 0,5891 |
| Recall | 0,1867 |
| F1-Score | 0,2836 |
| ROC-AUC | 0,7748 |

El baseline alcanzó un ROC-AUC de 0,7748, pero mostró un Recall de 0,1867 para la clase churn. Esto significa que, bajo el umbral utilizado, identifica una proporción limitada de los clientes que realmente abandonaron. Por esta razón, la Regresión Logística funciona como punto de referencia para evaluar si los modelos posteriores ofrecen una mejora en discriminación y recuperación de la clase positiva.

### Interpretabilidad del baseline

| Variable | Odds Ratio | IC OR 95% | Lectura |
|---|---|---|---|
| Age | 2,0928 | 1,9760–2,2219 | Asociación positiva con los odds de churn |
| Geography = Germany | 2,2811 | 1,9671–2,6451 | Asociación positiva con los odds de churn |
| Balance | 1,1741 | 1,0942–1,2597 | Asociación positiva con los odds de churn |
| IsActiveMember | 0,5968 | 0,5605–0,6355 | Asociación negativa con los odds de churn |
| Gender = Male | 0,5917 | 0,5252–0,6666 | Asociación negativa con los odds de churn |
| CreditScore | 0,9175 | 0,8648–0,9735 | Asociación negativa con los odds de churn |

Los intervalos muestran qué estimaciones tienen evidencia estadística compatible con una asociación distinta de un Odds Ratio igual a 1. Estos resultados describen asociaciones dentro del modelo; no deben interpretarse como relaciones causales.

En particular, Age y Geography = Germany presentan odds superiores a 1, mientras que IsActiveMember y Gender = Male presentan odds inferiores a 1. La interpretación debe mantenerse en términos de asociación y del resto de variables incluidas en el modelo.

## Modelado supervisado, optimización y comparación

El segundo avance amplió el baseline mediante modelos basados en árboles y boosting. Se utilizaron Random Forest, XGBoost, XGBoost optimizado, LightGBM y CatBoost. La evaluación se realizó con Accuracy, Precision, Recall, F1-Score, ROC-AUC y PR-AUC. La validación cruzada se configuró con StratifiedKFold de cinco folds y se utilizó ROC-AUC como criterio para el Grid Search de XGBoost.

### Resultados comparativos de discriminación

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| Regresión Logística | 0,7748 | 0,4790 |
| Random Forest | 0,849 | 0,674 |
| XGBoost | 0,860 | 0,708 |
| XGBoost Optimizado | 0,869 | 0,716 |
| LightGBM | 0,861 | 0,703 |
| CatBoost | 0,868 | 0,725 |
| Stacking | 0,870 | 0,724 |

Los valores de ROC-AUC y PR-AUC corresponden a las curvas generadas en el Avance 2. La mejora frente al baseline es consistente: los modelos de boosting alcanzan una mayor capacidad de discriminación y el Stacking obtiene el ROC-AUC más alto entre los resultados disponibles, con 0,870. CatBoost presenta el PR-AUC más alto de la comparación, con 0,725.

### XGBoost optimizado

La búsqueda de hiperparámetros evaluó learning_rate, max_depth y n_estimators. La configuración seleccionada fue learning_rate=0,05, max_depth=3 y n_estimators=300. El ROC-AUC promedio de validación cruzada registrado para esa configuración fue 0,8657; sobre el conjunto de prueba obtuvo Accuracy 0,8695, Precision 0,7874, Recall 0,4914, F1-Score 0,6051, ROC-AUC 0,8689 y PR-AUC 0,7201.

El resultado muestra que la optimización produjo una mejora respecto a XGBoost base en el conjunto de prueba, especialmente en las métricas de discriminación. Sin embargo, el Recall sigue por debajo de 0,50, por lo que la capacidad para recuperar todos los clientes con churn continúa siendo un aspecto relevante.

### Importancia de variables

> **[Imagen del PDF: gráfico de barras horizontales "Top 10 Feature Importance - XGBoost Optimizado" (eje X: Importancia (%)). Orden de variables, de mayor a menor: NumOfProducts, IsActiveMember, Age, Geography = Germany, Gender = Male, Balance, HasCrCard, CreditScore, Tenure, EstimatedSalary.]**

Top 10 de variables según la importancia del XGBoost optimizado. La importancia es relativa al modelo y no implica causalidad.

En el XGBoost optimizado, NumOfProducts, Age e IsActiveMember concentran las mayores importancias relativas. Geography = Germany también destaca frente a otras variables. La lectura correcta es que estas variables contribuyeron más al proceso predictivo de este modelo; no significa que por sí solas causen el churn.

### Evaluación final y Stacking

El Stacking combina las predicciones probabilísticas de XGBoost, LightGBM y CatBoost y utiliza una Regresión Logística como meta-learner. La arquitectura emplea validación cruzada interna de cinco folds y predict_proba como entrada al nivel superior.

> **[Imagen del PDF: "Matriz de Confusion - Stacking". Real No Churn: 1540 (predicho No Churn) / 53 (predicho Churn). Real Churn: 205 (predicho No Churn) / 202 (predicho Churn).]**

Matriz de confusión del Stacking sobre el conjunto de prueba.

La matriz contiene 1.540 verdaderos negativos, 53 falsos positivos, 203 falsos negativos y 204 verdaderos positivos. A partir de estos valores, el modelo alcanza Accuracy de 0,8720, Precision de 0,7938, Recall de 0,5012 y F1-Score de 0,6145. La curva ROC registra AUC de 0,870 y la curva Precision-Recall registra PR-AUC de 0,724.

La matriz permite observar una característica importante: aunque el modelo mejora notablemente frente al baseline en la recuperación de churn, todavía existen 203 clientes con churn que fueron clasificados como no churn. Por ello, la selección de un umbral operativo dependería del objetivo de negocio y del costo relativo de los falsos negativos y falsos positivos.

### Curvas de evaluación

La comparación de curvas ROC y Precision-Recall fue utilizada durante el Avance 2 para evaluar el comportamiento de los modelos en distintos umbrales. En la documentación se conservan los valores de ROC-AUC y PR-AUC en la tabla comparativa, evitando incorporar una figura multicolor al documento final. La lectura conjunta de ambas métricas es especialmente pertinente porque la clase churn representa aproximadamente una quinta parte de la base.

## Segmentación no supervisada

El tercer avance utiliza diez variables originales para clustering: seis variables continuas, dos binarias y dos categóricas. Geography y Gender se transformaron mediante OneHotEncoder; las variables continuas fueron estandarizadas y posteriormente se realizó una estandarización final de todas las variables que ingresaron al clustering. Exited quedó fuera de la construcción de los clusters y se utilizó únicamente después para analizar las tasas de churn.

### Selección de K-Means

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 97.252,4 | 0,1325 |
| 3 | 88.348,9 | 0,1336 |
| 4 | 82.585,6 | 0,1172 |
| 5 | 79.247,0 | 0,1005 |
| 6 | 76.404,9 | 0,1121 |
| 7 | 73.819,8 | 0,1102 |
| 8 | 71.917,6 | 0,1111 |

El mejor Silhouette dentro del rango evaluado se obtuvo con K=3, con 0,1336. Aunque la Inertia continúa disminuyendo al aumentar K, la selección se realizó priorizando el máximo Silhouette observado, evitando agregar segmentos sin una mejora equivalente en separación interna.

### K-Means y perfil de segmentos

| Cluster | Clientes | Edad | Balance | CreditScore | Productos | Activos | Tarjeta |
|---|---|---|---|---|---|---|---|
| 0 | 2.509 | 39,77 | 119.730 | 651,45 | 1,52 | 49,74% | 71,38% |
| 1 | 5.014 | 38,51 | 62.093 | 649,67 | 1,53 | 51,68% | 70,66% |
| 2 | 2.477 | 38,89 | 61.818 | 651,33 | 1,54 | 52,97% | 69,48% |

El Cluster 0 concentra 2.509 clientes y se diferencia principalmente por su balance promedio, de aproximadamente 119.730, frente a cerca de 62.000 en los otros dos segmentos. Su tasa de churn alcanza 32,44%, casi el doble de las tasas observadas en los Clusters 1 y 2. El Cluster 1 es el grupo más grande y presenta la menor tasa de churn, 16,15%.

> **[Imagen del PDF: gráfico de barras "Tasa de churn por cluster - K-Means" (eje X: Cluster 0, 1, 2; eje Y: Tasa de churn (%)). Cluster 0 ≈ 32%, Clusters 1 y 2 ≈ 16%.]**

Tasa de churn por segmento K-Means. Exited no intervino en la formación de los clusters.

La visualización mediante PCA permite representar los tres grupos en dos dimensiones, pero PC1 y PC2 explican conjuntamente 25,37% de la varianza. Por tanto, la separación visual debe interpretarse como una proyección del espacio original y no como una representación completa.

> **[Imagen del PDF: dispersión "K-Means visualizado mediante PCA" (eje X: Componente principal 1; eje Y: Componente principal 2), con los clusters 0, 1 y 2 en distintos colores.]**

K-Means proyectado en las dos primeras componentes principales.

## Reducción de dimensionalidad y DBSCAN

El PCA completo mostró que PC1 y PC2 explican 15,04% y 10,33% de la varianza, respectivamente, para un acumulado de 25,37%. Las tres primeras componentes alcanzan 35,32%. Para DBSCAN se utilizaron diez componentes principales, suficientes para alcanzar una varianza acumulada de 95,68%. t-SNE se aplicó sobre una muestra de 3.000 registros para facilitar una visualización alternativa de los patrones.

### Varianza explicada por PCA

| Componente | Varianza explicada | Acumulada |
|---|---|---|
| PC1 | 15,04% | 15,04% |
| PC2 | 10,33% | 25,37% |
| PC3 | 9,95% | 35,32% |
| PC4 | 9,30% | 44,63% |
| PC5 | 9,16% | 53,78% |
| PC6 | 9,08% | 62,87% |
| PC7 | 9,00% | 71,86% |
| PC8 | 8,83% | 80,69% |
| PC9 | 8,09% | 88,78% |
| PC10 | 6,89% | 95,68% |
| PC11 | 4,32% | 100,00% |

DBSCAN se evaluó con eps entre 0,7 y 1,5 y min_samples de 5, 10, 15 y 20. El criterio de selección consideró configuraciones con al menos dos clusters, priorizó aquellas con menos de 50% de ruido y, dentro de ese conjunto, eligió la de mayor Silhouette.

### Configuraciones relevantes de DBSCAN

| eps | min_samples | Clusters | Ruido | Silhouette |
|---|---|---|---|---|
| 1,2 | 20 | 28 | 67,00% | 0,1710 |
| 1,3 | 20 | 31 | 52,19% | 0,1889 |
| 1,4 | 20 | 34 | 38,53% | 0,1751 |
| 1,5 | 20 | 26 | 28,34% | 0,0908 |

La configuración seleccionada fue eps=1,4 y min_samples=20. Encontró 34 clusters y clasificó 3.853 registros como ruido, equivalentes al 38,53%. Su Silhouette fue 0,1751. La configuración eps=1,3 presentó un Silhouette mayor, 0,1889, pero superó el límite de 50% de ruido definido por el criterio del proyecto; por ello no fue seleccionada.

### Comparación de métodos

| Método | Clusters | Silhouette | Ruido |
|---|---|---|---|
| K-Means | 3 | 0,1336 | 0% |
| DBSCAN | 34 | 0,1751 | 38,53% |

DBSCAN obtiene un Silhouette superior, pero lo hace con una segmentación mucho más fragmentada y con una proporción considerable de ruido. K-Means produce tres segmentos más simples de comunicar y cubre todos los clientes. Por ello, ambos resultados son complementarios: DBSCAN detecta estructura local y puntos atípicos, mientras K-Means ofrece una segmentación global más manejable.

> **[Imagen del PDF: gráfico de barras "Comparación del Silhouette Score" (K-Means ≈ 0,133; DBSCAN ≈ 0,175).]**

Comparación de Silhouette Score entre K-Means y DBSCAN.

> **[Imagen del PDF: dispersión "DBSCAN visualizado mediante PCA" (eje X: Componente principal 1; eje Y: Componente principal 2), con leyenda de clusters desde -1 hasta 33.]**

Visualización de DBSCAN mediante una proyección PCA. La figura utiliza marcadores en blanco y negro para mantener el criterio visual del documento.

## Relación entre segmentación y churn

Una vez construidas las etiquetas, Exited se utilizó exclusivamente para medir la tasa de churn dentro de cada segmento. Esto permite estudiar si determinados grupos concentran una mayor proporción de abandono sin confundir segmentación no supervisada con predicción supervisada.

K-Means muestra una diferencia clara: el Cluster 0 alcanza 32,44% de churn, mientras el Cluster 1 registra 16,15% y el Cluster 2 16,67%. La diferencia entre el mayor y el menor segmento es de 16,29 puntos porcentuales. El patrón coincide con la distribución geográfica: el Cluster 0 corresponde a clientes de Germany, el Cluster 1 a France y el Cluster 2 a Spain.

DBSCAN produce una estructura mucho más granular. El grupo de ruido (-1) contiene 3.853 clientes y presenta 29,54% de churn. Entre los clusters identificados, el Cluster 9 registra la mayor tasa, 53,09%, seguido por el Cluster 7 con 45,83% y el Cluster 6 con 35,14%. Estas tasas deben interpretarse como descripciones de los grupos obtenidos y no como evidencia de causalidad.

| Grupo DBSCAN | Clientes | Churn | Tasa |
|---|---|---|---|
| 9 | 243 | 129 | 53,09% |
| 7 | 216 | 99 | 45,83% |
| 6 | 148 | 52 | 35,14% |
| 16 | 327 | 99 | 30,28% |
| -1 · ruido | 3.853 | 1.138 | 29,54% |
| 8 | 478 | 111 | 23,22% |
| 24 | 138 | 30 | 21,74% |

La variable Customer_Segment se incorporó al dataset final como una feature derivada del proceso de clustering. El dataset segmentado conserva las etiquetas KMeans_Cluster y DBSCAN_Cluster y añade Customer_Segment para facilitar su utilización posterior en análisis o procesos de negocio.

Es importante conservar la distinción metodológica: los clusters se construyeron sin Exited; las tasas de churn se calcularon después, como una capa de análisis sobre las etiquetas resultantes.

## Reproducibilidad y estructura de archivos

La ejecución fue organizada en tres scripts principales. Cada uno trabaja sobre el mismo dataset de entrada y genera resultados que respaldan la etapa correspondiente. El Avance 4 documenta estos componentes para que otra persona pueda reconstruir el flujo sin depender de una explicación oral.

| Archivo | Función |
|---|---|
| Churn_Modelling.csv | Dataset de entrada con 10.000 registros y 14 variables. |
| 1_EDA_RegresionLogistica.py | EDA, preparación, división train/test, Regresión Logística baseline, métricas, matriz de confusión, coeficientes, Odds Ratio e intervalos de confianza. |
| 2_GradientBoosting_Optimizacion.py | Comparación de modelos, validación cruzada, Grid Search de XGBoost, feature importance, Stacking, matriz de confusión y curvas ROC/PR. |
| 3_AprendizajeNoSupervisado.py | Preparación para clustering, K-Means, PCA, perfiles, DBSCAN, t-SNE, relación con churn y dataset segmentado. |

### Principales resultados exportados

Los scripts generan archivos CSV y PNG como respaldo. En el Avance 1 se exportan resultados_metricas.csv, coeficientes_modelo.csv, odds_ratios_modelo.csv e intervalos_confianza.csv.

En el Avance 2 se generan, entre otros, comparacion_final.csv, resultados_modelos.csv, resultados_cross_validation.csv, resultados_cross_validation_stacking.csv, mejores_parametros_xgboost.csv, resultado_stacking.csv, archivos de feature importance, matriz_confusion_mejor_modelo.csv, comparacion_curvas_roc.png y comparacion_curvas_pr.png.

En el Avance 3 se generan evaluacion_kmeans.csv, cantidad_clientes_kmeans.csv, centroides_kmeans.csv, perfil_numerico_kmeans.csv, perfil_geografia_kmeans.csv, perfil_genero_kmeans.csv, churn_por_cluster.csv, evaluacion_dbscan.csv, churn_por_cluster_dbscan.csv, datos_preprocesados.csv, dataset_segmentado.csv, clientes_con_segmentacion.csv, resultado_tsne.csv y las visualizaciones asociadas.

### Dependencias utilizadas

El proyecto utiliza Python y librerías de análisis y Machine Learning como pandas, NumPy, Matplotlib, Seaborn, tabulate, scikit-learn, statsmodels, XGBoost, LightGBM y CatBoost. Los scripts contienen la configuración de random_state=42 para mantener la reproducibilidad de los principales procesos aleatorios.

## Conclusiones

El proyecto construyó una solución analítica progresiva para el problema de churn bancario. El análisis exploratorio permitió establecer la calidad y estructura del dataset y mostró que 20,37% de los clientes presentan churn. La Regresión Logística sirvió como baseline y evidenció una capacidad de discriminación de 0,7748, pero un Recall bajo para la clase positiva.

La segunda etapa mostró que los modelos de boosting mejoran sustancialmente la discriminación. XGBoost optimizado alcanzó ROC-AUC de 0,8689 y PR-AUC de 0,7201 sobre prueba, mientras que el Stacking obtuvo ROC-AUC de 0,870 y PR-AUC de 0,724 en las curvas comparativas. La matriz de confusión del Stacking confirma una mejora en la recuperación de churn frente al baseline, aunque todavía existen falsos negativos relevantes.

La tercera etapa aportó una perspectiva diferente. K-Means permitió construir tres segmentos interpretables y mostró que el Cluster 0 presenta 32,44% de churn, frente a 16,15% y 16,67% en los otros segmentos. DBSCAN encontró una estructura más granular, con 34 clusters y 38,53% de registros considerados ruido. Su Silhouette fue superior al de K-Means, pero la segmentación resultó más fragmentada y menos sencilla de comunicar.

En conjunto, los resultados no deben reducirse a escoger un único algoritmo. La clasificación supervisada permite estimar la probabilidad de pertenencia a la clase churn, mientras la segmentación permite describir grupos de clientes y observar cómo cambia la tasa de abandono entre ellos. La combinación de ambas perspectivas proporciona una base más completa para análisis posteriores.

El resultado final del Avance 4 es una documentación trazable del proyecto ya construido: identifica los datos, explica la metodología, conserva las decisiones principales, integra los resultados y deja organizada la ruta de ejecución. La solución queda preparada para continuar con análisis de negocio, ajuste de umbrales, evaluación de costos de error o explotación de los segmentos, siempre diferenciando claramente entre predicción, asociación y segmentación.

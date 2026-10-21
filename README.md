# PROYECTO INTEGRADOR — CHURN BANCARIO
## AVANCE 4 — DOCUMENTACIÓN Y CONSOLIDACIÓN

### Descripción del proyecto
FinanceGuard es un banco digital que ha experimentado un crecimiento exponencial en los últimos años, alcanzando 10,000 clientes activos. Sin embargo, la dirección ha detectado un incremento preocupante en la tasa de abandono de clientes (churn), que actualmente alcanza el 20% anual, resultando en pérdidas millonarias.

Primero se comprendió la estructura y calidad de los datos; después se construyeron y compararon modelos de clasificación para estimar el churn; finalmente se aplicaron técnicas de aprendizaje no supervisado para segmentar clientes y estudiar cómo los segmentos se relacionan con la variable objetivo.

El Avance 4 no incorpora un nuevo modelo ni pretende demostrar un análisis adicional. Su función es consolidar lo que ya fue ejecutado, explicar las decisiones tomadas, presentar los resultados relevantes y dejar una ruta clara para reproducir o continuar el proyecto. La documentación se limita a procedimientos y resultados respaldados por los scripts, el dataset y los resultados disponibles.

### Problema abordado
La variable `Exited` representa si un cliente abandonó la entidad. El problema combina dos necesidades complementarias: identificar qué tan bien pueden distinguirse clientes con y sin churn mediante modelos supervisados, y descubrir grupos de clientes con características similares mediante clustering. La segmentación no sustituye la predicción; funciona como una perspectiva adicional para interpretar la heterogeneidad de la base.

### Objetivo
Como Científico de Datos Junior, tu misión es desarrollar un modelo de Machine Learning completo que prediga qué clientes tienen mayor probabilidad de abandonar el banco. Este modelo permitirá al equipo de retención implementar estrategias personalizadas para retener a los clientes en riesgo, con el objetivo de reducir el churn al 15%.

### Alcance del Avance 4
La documentación reúne el flujo completo desde el dataset de entrada hasta el dataset segmentado. Incluye preparación de datos, división entrenamiento/prueba, validación cruzada, modelos supervisados, optimización de XGBoost, interpretabilidad de la regresión logística, comparación de modelos, K-Means, DBSCAN, PCA, t-SNE, relación entre segmentos y churn, exportación de resultados y reproducibilidad.

---

## Datos y preparación
El archivo de entrada utilizado fue `Churn_Modelling.csv`. Contiene 10.000 registros y 14 variables. La validación realizada en el Avance 3 confirmó que las variables analizadas no presentan valores nulos y que no existen filas duplicadas. La misma base fue utilizada como referencia para los procesos de clasificación y clustering.

### Variables del dataset

| Variable | Tipo | Uso principal |
| :--- | :--- | :--- |
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

En los modelos supervisados se eliminaron `RowNumber`, `CustomerId` y `Surname` y se utilizó `Exited` como variable objetivo. Las variables numéricas se estandarizaron mediante `StandardScaler` y las categóricas mediante `OneHotEncoder` con eliminación de la primera categoría. En el clustering se mantuvieron las variables binarias en escala 0–1 antes de la estandarización final y también se aplicó `OneHotEncoder` a `Geography` y `Gender`.

---

## Distribución inicial del churn
La base contiene 7.963 clientes sin churn y 2.037 con churn, equivalentes a 79,63% y 20,37%, respectivamente. Esta distribución confirma que la clase churn es minoritaria, por lo que la evaluación no debe depender únicamente de *Accuracy*.

### Continuidad de los tres avances
El proyecto debe entenderse como un único flujo analítico. Cada avance respondió a una etapa diferente del problema y los resultados de una etapa sirven de base para la siguiente.

* **Comprensión y baseline:** El primer avance se concentró en el análisis exploratorio, la calidad del dataset, la clasificación de variables, la preparación de los datos y la construcción de una Regresión Logística como modelo baseline. También se estudiaron correlaciones, multicolinealidad mediante VIF, coeficientes, Odds Ratio e intervalos de confianza.
* **Mejora de la predicción:** El segundo avance amplió el modelado con Random Forest, XGBoost, XGBoost optimizado, LightGBM y CatBoost. Se incorporó validación cruzada estratificada de cinco folds, Grid Search para XGBoost, comparación de métricas y un modelo de Stacking formado por XGBoost, LightGBM y CatBoost con Regresión Logística como meta-learner.
* **Segmentación de clientes:** El tercer avance trasladó el análisis al aprendizaje no supervisado. Se evaluó el número de clusters para K-Means mediante Inertia y Silhouette, se construyeron tres segmentos, se perfiló cada grupo y se compararon sus tasas de churn. Después se evaluó DBSCAN con distintas combinaciones de eps y min_samples. PCA y t-SNE se utilizaron como herramientas de reducción de dimensionalidad y visualización.

---

## Análisis exploratorio y modelo baseline
La división train/test se realizó con 80% de los datos para entrenamiento y 20% para prueba, usando `random_state=42` y estratificación para conservar la proporción de churn.

### Conjuntos de Datos
* **Entrenamiento:** 8.000 registros (80%) | Churn: 1.630 | No churn: 6.370
* **Prueba:** 2.000 registros (20%) | Churn: 407 | No churn: 1.593
* **Total:** 10.000 registros (100%) | Churn: 2.037 | No churn: 7.963

### Regresión Logística baseline
* **Accuracy:** 0,8080
* **Precision:** 0,5891
* **Recall:** 0,1867
* **F1-Score:** 0,2836
* **ROC-AUC:** 0,7748

El baseline alcanzó un ROC-AUC de 0,7748, pero mostró un Recall de 0,1867 para la clase churn. Esto significa que identifica una proporción limitada de los clientes que realmente abandonaron.

### Interpretabilidad del baseline

| Variable | Odds Ratio | IC OR 95% | Lectura |
| :--- | :---: | :---: | :--- |
| Age | 2,0928 | 1,9760–2,2219 | Asociación positiva con los odds de churn |
| Geography = Germany | 2,2811 | 1,9671–2,6451 | Asociación positiva con los odds de churn |
| Balance | 1,1741 | 1,0942–1,2597 | Asociación positiva con los odds de churn |
| IsActiveMember | 0,5968 | 0,5605–0,6355 | Asociación negativa con los odds de churn |
| Gender = Male | 0,5917 | 0,5252–0,6666 | Asociación negativa con los odds de churn |
| CreditScore | 0,9175 | 0,8648–0,9735 | Asociación negativa con los odds de churn |

---

## Modelado supervisado, optimización y comparación

### Resultados comparativos de discriminación

| Modelo | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| Regresión Logística | 0,7748 | 0,4790 |
| Random Forest | 0,849 | 0,674 |
| XGBoost | 0,860 | 0,708 |
| XGBoost Optimizado | 0,869 | 0,716 |
| LightGBM | 0,861 | 0,703 |
| CatBoost | 0,868 | 0,725 |
| Stacking | 0,870 | 0,724 |

### XGBoost optimizado
La configuración seleccionada fue `learning_rate=0.05`, `max_depth=3` y `n_estimators=300`. El ROC-AUC promedio de validación cruzada fue 0,8657; sobre el conjunto de prueba obtuvo:
* **Accuracy:** 0,8695
* **Precision:** 0,7874
* **Recall:** 0,4914
* **F1-Score:** 0,6051
* **ROC-AUC:** 0,8689
* **PR-AUC:** 0,7201

### Evaluación final y Stacking
El Stacking combina XGBoost, LightGBM y CatBoost con una Regresión Logística como meta-learner.
* **Matriz de confusión (Prueba):** 1.540 verdaderos negativos, 53 falsos positivos, 203 falsos negativos y 204 verdaderos positivos.
* **Métricas:** Accuracy de 0,8720, Precision de 0,7938, Recall de 0,5012 y F1-Score de 0,6145.

---

## Segmentación no supervisada

### Selección de K-Means

| K | Inertia | Silhouette |
| :-: | :---: | :---: |
| 2 | 97.252,4 | 0,1325 |
| 3 | 88.348,9 | 0,1336 |
| 4 | 82.585,6 | 0,1172 |
| 5 | 79.247,0 | 0,1005 |
| 6 | 76.404,9 | 0,1121 |
| 7 | 73.819,8 | 0,1102 |
| 8 | 71.917,6 | 0,1111 |

El mejor Silhouette Score se obtuvo con `K=3` (0,1336).

### K-Means y perfil de segmentos

| Cluster | Clientes | Edad | Balance | CreditScore | Productos | Activos | Tarjeta | Tasa Churn |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | 2.509 | 39,77 | 119.730 | 651,45 | 1,52 | 49,74% | 71,38% | **32,44%** |
| 1 | 5.014 | 38,51 | 62.093 | 649,67 | 1,53 | 51,68% | 70,66% | **16,15%** |
| 2 | 2.477 | 38,89 | 61.818 | 651,33 | 1,54 | 52,97% | 69,48% | **16,67%** |

### Reducción de dimensionalidad y DBSCAN
Para DBSCAN se utilizaron diez componentes principales de PCA (95,68% de varianza acumulada). La configuración elegida fue `eps=1.4` y `min_samples=20`, encontrando 34 clusters con un 38,53% de ruido y Silhouette de 0,1751.

### Comparación de métodos de clustering

| Método | Clusters | Silhouette | Ruido |
| :--- | :---: | :---: | :---: |
| K-Means | 3 | 0,1336 | 0% |
| DBSCAN | 34 | 0,1751 | 38,53% |

---

## Relación entre segmentación y churn

### Grupos DBSCAN con mayor tasa de Churn

| Grupo DBSCAN | Clientes | Churn | Tasa |
| :---: | :---: | :---: | :---: |
| 9 | 243 | 129 | 53,09% |
| 7 | 216 | 99 | 45,83% |
| 6 | 148 | 52 | 35,14% |
| 16 | 327 | 99 | 30,28% |
| -1 (Ruido) | 3.853 | 1.138 | 29,54% |
| 8 | 478 | 111 | 23,22% |
| 24 | 138 | 30 | 21,74% |

---

## Reproducibilidad y estructura de archivos

| Archivo | Función |
| :--- | :--- |
| `Churn_Modelling.csv` | Dataset de entrada con 10.000 registros y 14 variables. |
| `1_EDA_RegresionLogistica.py` | EDA, preparación, división train/test, Regresión Logística baseline, métricas, Odds Ratio. |
| `2_GradientBoosting_Optimizacion.py` | Comparación de modelos, validación cruzada, Grid Search de XGBoost, Stacking. |
| `3_AprendizajeNoSupervisado.py` | Preparación para clustering, K-Means, PCA, DBSCAN, t-SNE y dataset segmentado. |

---

## Conclusiones
El proyecto construyó una solución analítica progresiva. La segunda etapa mostró que los modelos de boosting mejoran sustancialmente la discriminación, donde el Stacking obtuvo el mejor desempeño (ROC-AUC de 0,870). La tercera etapa aportó una perspectiva diferente mediante clustering, donde K-Means identificó un grupo de alto riesgo (Cluster 0) con 32,44% de churn correlacionado con clientes en Alemania. La combinación de ambas perspectivas proporciona una base completa para las estrategias de retención del banco.
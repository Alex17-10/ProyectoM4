# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from IPython.display import display
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Métricas
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Configuración
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)


# Cargar datos
ruta = Path(__file__).parent / "Churn_Modelling.csv"
df = pd.read_csv(ruta)

# Exploración inicial


print("1. DIMENSIONES DEL DATASET")


dimensiones = [
    ["Filas", f"{df.shape[0]:,}"],
    ["Columnas", f"{df.shape[1]:,}"]
]

print(tabulate(
    dimensiones,
    headers=["Característica", "Cantidad"],
    tablefmt="grid"
))


print("\n" + "=" * 80)
print("2. PRIMERAS 5 FILAS")
print("=" * 80)

print(df.head().to_string(index=False))


print("\n" + "=" * 80)
print("3. INFORMACIÓN GENERAL")
print("=" * 80)

info_df = pd.DataFrame({
    "Variable": df.columns,
    "Tipo": df.dtypes.astype(str).values,
    "No nulos": df.notna().sum().values,
    "Faltantes": df.isnull().sum().values,
    "Únicos": df.nunique().values
})

print(tabulate(
    info_df,
    headers="keys",
    tablefmt="grid",
    showindex=False
))


print("\n" + "=" * 80)
print("4. VALORES FALTANTES")
print("=" * 80)

missing_df = pd.DataFrame({
    "Variable": df.columns,
    "Faltantes": df.isnull().sum().values,
    "Porcentaje": (
        df.isnull().sum().values / len(df) * 100
    ).round(2)
})

print(tabulate(
    missing_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".2f"
))


print("\n" + "=" * 80)
print("5. REGISTROS DUPLICADOS")
print("=" * 80)

duplicados = df.duplicated().sum()

duplicados_df = pd.DataFrame({
    "Indicador": ["Filas duplicadas"],
    "Cantidad": [duplicados]
})

print(tabulate(
    duplicados_df,
    headers="keys",
    tablefmt="grid",
    showindex=False
))


print("\n" + "=" * 80)
print("6. ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 80)

estadisticas = df.describe().T.reset_index()

estadisticas.columns = [
    "Variable",
    "Cantidad",
    "Media",
    "Desv. Est.",
    "Mínimo",
    "25%",
    "Mediana",
    "75%",
    "Máximo"
]

print(tabulate(
    estadisticas,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".2f"
))


print("\n" + "=" * 80)
print("7. VARIABLES CATEGÓRICAS")
print("=" * 80)

categorical_columns = [
    column
    for column in df.columns
    if df[column].dtype.name in ["str", "object"]
]

categorical_rows = []

for column in categorical_columns:

    cantidad = df[column].nunique()

    valores = df[column].dropna().unique()

    if cantidad <= 10:
        valores_texto = ", ".join(
            str(valor) for valor in valores
        )
    else:
        valores_texto = f"{cantidad:,} categorías"

    categorical_rows.append([
        column,
        cantidad,
        valores_texto
    ])

categorical_df = pd.DataFrame(
    categorical_rows,
    columns=[
        "Variable",
        "Categorías",
        "Valores"
    ]
)

print(tabulate(
    categorical_df,
    headers="keys",
    tablefmt="grid",
    showindex=False
))


print("\n" + "=" * 80)
print("8. DISTRIBUCIÓN DE CHURN")
print("=" * 80)

churn_counts = df["Exited"].value_counts().sort_index()

churn_df = pd.DataFrame({
    "Estado": [
        "No abandonó",
        "Abandonó"
    ],
    "Exited": [0, 1],
    "Cantidad": [
        churn_counts.get(0, 0),
        churn_counts.get(1, 0)
    ],
    "Porcentaje": [
        churn_counts.get(0, 0) / len(df) * 100,
        churn_counts.get(1, 0) / len(df) * 100
    ]
})

print(tabulate(
    churn_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".2f"
))


print("\n" + "=" * 80)
print("9. GRÁFICO DE CHURN")
print("=" * 80)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Exited"
)

plt.title("Distribución de clientes según Churn")
plt.xlabel("Estado del cliente")
plt.ylabel("Cantidad de clientes")

plt.xticks(
    [0, 1],
    ["No abandonó", "Abandonó"]
)

plt.show()


print("\n" + "=" * 80)
print("10. VARIABLES PARA EL MODELO")
print("=" * 80)

columns_to_drop = [
    "RowNumber",
    "CustomerId",
    "Surname"
]

X = df.drop(
    columns=columns_to_drop + ["Exited"]
)

y = df["Exited"]

variables_modelo = pd.DataFrame({
    "Variable": X.columns,
    "Tipo": X.dtypes.astype(str).values
})

print(tabulate(
    variables_modelo,
    headers="keys",
    tablefmt="grid",
    showindex=False
))


print("\n" + "=" * 80)
print("11. CLASIFICACIÓN DE VARIABLES")
print("=" * 80)

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = [
    column
    for column in X.columns
    if X[column].dtype.name in ["str", "object"]
]

tipos_variables = pd.DataFrame({
    "Tipo": [
        "Numéricas",
        "Categóricas"
    ],
    "Cantidad": [
        len(numeric_features),
        len(categorical_features)
    ],
    "Variables": [
        ", ".join(numeric_features),
        ", ".join(categorical_features)
    ]
})

print(tabulate(
    tipos_variables,
    headers="keys",
    tablefmt="grid",
    showindex=False
))

print("\n" + "=" * 80)
print("12. PREPARACIÓN DE DATOS")
print("=" * 80)

# Separación de variables predictoras y objetivo
columns_to_drop = [
    "RowNumber",
    "CustomerId",
    "Surname"
]

X = df.drop(columns=columns_to_drop + ["Exited"])
y = df["Exited"]

print("Variables predictoras:")
print(tabulate(
    [[column] for column in X.columns],
    headers=["Variable"],
    tablefmt="grid"
))

print("\nVariable objetivo:")
print("Exited")


# Identificación de variables
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = [
    column
    for column in X.columns
    if X[column].dtype.name in ["str", "object"]
]

preparacion_df = pd.DataFrame({
    "Tipo": ["Numéricas", "Categóricas"],
    "Cantidad": [
        len(numeric_features),
        len(categorical_features)
    ],
    "Variables": [
        ", ".join(numeric_features),
        ", ".join(categorical_features)
    ]
})

print("\nClasificación de variables:")
print(tabulate(
    preparacion_df,
    headers="keys",
    tablefmt="grid",
    showindex=False
))


# Train / Test Split
print("\n" + "=" * 80)
print("13. DIVISIÓN TRAIN / TEST")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

split_df = pd.DataFrame({
    "Conjunto": [
        "Entrenamiento",
        "Prueba"
    ],
    "Registros": [
        len(X_train),
        len(X_test)
    ],
    "Porcentaje": [
        len(X_train) / len(X) * 100,
        len(X_test) / len(X) * 100
    ],
    "Churn (1)": [
        y_train.sum(),
        y_test.sum()
    ],
    "No Churn (0)": [
        (y_train == 0).sum(),
        (y_test == 0).sum()
    ]
})

print(tabulate(
    split_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".2f"
))


# Preprocesamiento
print("\n" + "=" * 80)
print("14. PREPROCESAMIENTO")
print("=" * 80)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

print("\n" + "=" * 80)
print("15. ANÁLISIS DE MULTICOLINEALIDAD")
print("=" * 80)

# Matriz de correlación
correlation_matrix = X[numeric_features].corr()

print("\nMatriz de correlación:")
print(tabulate(
    correlation_matrix.round(2),
    headers="keys",
    tablefmt="grid",
    showindex=True
))


# Correlaciones altas entre variables
print("\nCorrelaciones superiores a 0.70:")

high_correlations = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):

        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= 0.70:
            high_correlations.append([
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                round(correlation, 2)
            ])

if high_correlations:
    print(tabulate(
        high_correlations,
        headers=[
            "Variable 1",
            "Variable 2",
            "Correlación"
        ],
        tablefmt="grid"
    ))
else:
    print("No se encontraron correlaciones superiores a 0.70.")

print("\n" + "=" * 80)
print("16. FACTOR DE INFLACIÓN DE LA VARIANZA (VIF)")
print("=" * 80)

# Agregar constante para calcular correctamente el VIF
X_vif = add_constant(X[numeric_features])

vif_data = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
})

# Eliminar la constante de la tabla final
vif_data = vif_data[vif_data["Variable"] != "const"]

print(tabulate(
    vif_data.round(2),
    headers="keys",
    tablefmt="grid",
    showindex=False
))

# Interpretación
print("\nInterpretación del VIF:")

for _, fila in vif_data.iterrows():
    if fila["VIF"] < 5:
        estado = "Sin evidencia de multicolinealidad."
    elif fila["VIF"] < 10:
        estado = "Multicolinealidad moderada."
    else:
        estado = "Multicolinealidad alta."

    print(f"- {fila['Variable']}: VIF = {fila['VIF']:.2f} → {estado}")

print("\n" + "=" * 80)
print("17. REGRESIÓN LOGÍSTICA - MODELO BASELINE")
print("=" * 80)

# Crear pipeline
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

# Entrenamiento
logistic_model.fit(
    X_train,
    y_train
)

print("Modelo entrenado correctamente.")


# Predicciones
y_pred = logistic_model.predict(X_test)

y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]

print("\nPredicciones generadas correctamente.")

print(f"Clientes evaluados: {len(X_test):,}")
print(f"Predicciones de churn: {y_pred.sum():,}")
print(f"Predicciones de no churn: {(y_pred == 0).sum():,}")

print("\n" + "=" * 80)
print("18. EVALUACIÓN DEL MODELO")
print("=" * 80)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metricas_df = pd.DataFrame({
    "Métrica": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Resultado": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

print(tabulate(
    metricas_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))


# Reporte de clasificación
print("\n" + "=" * 80)
print("19. REPORTE DE CLASIFICACIÓN")
print("=" * 80)

print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "No abandonó",
        "Abandonó"
    ],
    digits=4
))


# Matriz de confusión
print("\n" + "=" * 80)
print("20. MATRIZ DE CONFUSIÓN")
print("=" * 80)

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=["Real: No abandonó", "Real: Abandonó"],
    columns=["Predicho: No abandonó", "Predicho: Abandonó"]
)

print(tabulate(
    cm_df,
    headers="keys",
    tablefmt="grid"
))


# Visualización de matriz de confusión
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No abandonó", "Abandonó"],
    yticklabels=["No abandonó", "Abandonó"]
)

plt.title("Matriz de confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Valor real")

plt.tight_layout()
plt.show()


# Curva ROC
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_pred_proba
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Regresión Logística (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Clasificador aleatorio"
)

plt.title("Curva ROC - Regresión Logística")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("21. FUNCIÓN SIGMOIDE")
print("=" * 80)

print("La regresión logística transforma una combinación lineal de")
print("las variables predictoras mediante la función sigmoide.")
print()
print("P(z) = 1 / (1 + e^(-z))")
print()
print("El resultado es una probabilidad entre 0 y 1.")
print("Probabilidad >= 0.50 → predicción de churn (1)")
print("Probabilidad < 0.50 → predicción de no churn (0)")


print("\n" + "=" * 80)
print("22. COEFICIENTES DEL MODELO")
print("=" * 80)

feature_names = logistic_model.named_steps[
    "preprocessor"
].get_feature_names_out()

clean_feature_names = []

for name in feature_names:
    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    if name == "Geography_Germany":
        name = "Geography = Germany"
    elif name == "Geography_Spain":
        name = "Geography = Spain"
    elif name == "Gender_Male":
        name = "Gender = Male"

    clean_feature_names.append(name)

coefficients = logistic_model.named_steps[
    "classifier"
].coef_[0]

coefficients_df = pd.DataFrame({
    "Variable": clean_feature_names,
    "Coeficiente": coefficients
})

coefficients_df = coefficients_df.sort_values(
    "Coeficiente",
    ascending=False
).reset_index(drop=True)

coefficients_df["Direccion"] = np.where(
    coefficients_df["Coeficiente"] > 0,
    "Mayor riesgo de churn",
    "Menor riesgo de churn"
)

print(tabulate(
    coefficients_df.round(4),
    headers="keys",
    tablefmt="grid",
    showindex=False
))


print("\n" + "=" * 80)
print("23. ODDS RATIOS")
print("=" * 80)

odds_ratios_df = coefficients_df.copy()

odds_ratios_df["Odds Ratio"] = np.exp(
    odds_ratios_df["Coeficiente"]
)

odds_ratios_df["Cambio en odds (%)"] = np.where(
    odds_ratios_df["Odds Ratio"] >= 1,
    (odds_ratios_df["Odds Ratio"] - 1) * 100,
    (1 - odds_ratios_df["Odds Ratio"]) * -100
)

interpretations = []

for _, row in odds_ratios_df.iterrows():

    variable = row["Variable"]
    odds_ratio = row["Odds Ratio"]

    if variable == "Geography = Germany":

        change = abs(odds_ratio - 1) * 100

        interpretation = (
            f"Los clientes de Alemania presentan "
            f"{change:.1f}% mayores odds de churn "
            "que los clientes de Francia."
        )

    elif variable == "Geography = Spain":

        change = abs(odds_ratio - 1) * 100

        if odds_ratio >= 1:
            interpretation = (
                f"Los clientes de Espana presentan "
                f"{change:.1f}% mayores odds de churn "
                "que los clientes de Francia."
            )
        else:
            interpretation = (
                f"Los clientes de Espana presentan "
                f"{change:.1f}% menores odds de churn "
                "que los clientes de Francia."
            )

    elif variable == "Gender = Male":

        change = abs(odds_ratio - 1) * 100

        if odds_ratio >= 1:
            interpretation = (
                f"Los hombres presentan "
                f"{change:.1f}% mayores odds de churn "
                "que las mujeres."
            )
        else:
            interpretation = (
                f"Los hombres presentan "
                f"{change:.1f}% menores odds de churn "
                "que las mujeres."
            )

    elif variable in [
        "HasCrCard",
        "IsActiveMember"
    ]:

        change = abs(odds_ratio - 1) * 100

        if odds_ratio >= 1:
            interpretation = (
                f"Pasar de 0 a 1 en esta variable "
                f"se asocia con {change:.1f}% mayores "
                "odds de churn."
            )
        else:
            interpretation = (
                f"Pasar de 0 a 1 en esta variable "
                f"se asocia con {change:.1f}% menores "
                "odds de churn."
            )

    else:

        change = abs(odds_ratio - 1) * 100

        if odds_ratio >= 1:
            interpretation = (
                f"Un aumento de 1 desviacion estandar "
                f"se asocia con {change:.1f}% mayores "
                "odds de churn."
            )
        else:
            interpretation = (
                f"Un aumento de 1 desviacion estandar "
                f"se asocia con {change:.1f}% menores "
                "odds de churn."
            )

    interpretations.append(interpretation)


odds_ratios_df["Interpretacion"] = interpretations

odds_ratios_df = odds_ratios_df[
    [
        "Variable",
        "Coeficiente",
        "Odds Ratio",
        "Cambio en odds (%)",
        "Direccion",
        "Interpretacion"
    ]
]

print(tabulate(
    odds_ratios_df.round(4),
    headers="keys",
    tablefmt="grid",
    showindex=False
))


print("\n" + "=" * 80)
print("24. INTERVALOS DE CONFIANZA")
print("=" * 80)

X_train_transformed = logistic_model.named_steps[
    "preprocessor"
].transform(X_train)

X_train_stats = sm.add_constant(
    X_train_transformed,
    has_constant="add"
)

stats_model = sm.Logit(
    y_train,
    X_train_stats
)

stats_result = stats_model.fit(
    disp=False
)

confidence_intervals = stats_result.conf_int()

stats_names = ["Intercept"] + clean_feature_names

ci_df = pd.DataFrame({
    "Variable": stats_names,
    "Coeficiente": stats_result.params,
    "IC inferior": confidence_intervals[0],
    "IC superior": confidence_intervals[1]
})

ci_df["Odds Ratio"] = np.exp(
    ci_df["Coeficiente"]
)

ci_df["IC OR inferior"] = np.exp(
    ci_df["IC inferior"]
)

ci_df["IC OR superior"] = np.exp(
    ci_df["IC superior"]
)

ci_df["Incluye OR = 1"] = np.where(
    (ci_df["IC OR inferior"] <= 1) &
    (ci_df["IC OR superior"] >= 1),
    "Si",
    "No"
)

ci_df["Evidencia estadistica"] = np.where(
    ci_df["Variable"] == "Intercept",
    "Parametro base del modelo",
    np.where(
        ci_df["Incluye OR = 1"] == "Si",
        "No concluyente",
        "Asociacion significativa"
    )
)

ci_df = ci_df[
    [
        "Variable",
        "Coeficiente",
        "Odds Ratio",
        "IC OR inferior",
        "IC OR superior",
        "Incluye OR = 1",
        "Evidencia estadistica"
    ]
]

print(tabulate(
    ci_df.round(4),
    headers="keys",
    tablefmt="grid",
    showindex=False
))


print("\n" + "=" * 80)
print("25. RESUMEN DEL MODELO BASELINE")
print("=" * 80)

resumen_df = pd.DataFrame({
    "Metrica": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Resultado": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

print(tabulate(
    resumen_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))

print("\nInterpretacion:")

print(
    f"- Accuracy: {accuracy:.4f}. "
    "Representa la proporcion total de predicciones correctas."
)

print(
    f"- Precision: {precision:.4f}. "
    "De los clientes predichos como churn, esta proporcion "
    "realmente abandono."
)

print(
    f"- Recall: {recall:.4f}. "
    "El modelo identifico correctamente esta proporcion "
    "de los clientes que realmente abandonaron."
)

print(
    f"- F1-Score: {f1:.4f}. "
    "Combina Precision y Recall en una sola metrica."
)

print(
    f"- ROC-AUC: {auc:.4f}. "
    "Indica una capacidad de discriminacion razonable "
    "entre clientes que abandonan y permanecen."
)

print("\nConclusion del baseline:")

print(
    "La regresion logistica presenta una capacidad de "
    "discriminacion razonable, pero su Recall para la clase "
    "churn es bajo. Por lo tanto, el modelo identifica una "
    "proporcion limitada de los clientes que realmente "
    "abandonan bajo el umbral de clasificacion utilizado."
)

print(
    "El modelo se establece como baseline para comparar "
    "posteriormente con modelos de Machine Learning mas "
    "complejos."
)


print("\n" + "=" * 80)
print("26. EXPORTACION DE RESULTADOS")
print("=" * 80)

carpeta_salida = Path(__file__).parent

metricas_df.to_csv(
    carpeta_salida / "resultados_metricas.csv",
    index=False,
    encoding="utf-8-sig"
)

coefficients_df.to_csv(
    carpeta_salida / "coeficientes_modelo.csv",
    index=False,
    encoding="utf-8-sig"
)

odds_ratios_df.to_csv(
    carpeta_salida / "odds_ratios_modelo.csv",
    index=False,
    encoding="utf-8-sig"
)

ci_df.to_csv(
    carpeta_salida / "intervalos_confianza.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Resultados exportados correctamente en:")
print(carpeta_salida)

print("\nArchivos creados:")
print("- resultados_metricas.csv")
print("- coeficientes_modelo.csv")
print("- odds_ratios_modelo.csv")
print("- intervalos_confianza.csv")


print("Preprocesamiento configurado correctamente.")

print("\nVariables numéricas:")
print(", ".join(numeric_features))

print("\nVariables categóricas:")
print(", ".join(categorical_features))

print("\nProcesamiento aplicado:")
print("- Variables numéricas → StandardScaler")
print("- Variables categóricas → OneHotEncoder")
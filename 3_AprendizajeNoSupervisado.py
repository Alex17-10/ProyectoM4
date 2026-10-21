from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# Configuración general

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
OUTPUT_DIR = BASE_DIR / "resultados_avance_3"

OUTPUT_DIR.mkdir(exist_ok=True)

for archivo in OUTPUT_DIR.iterdir():
    if archivo.is_file():
        archivo.unlink()

RANDOM_STATE = 42

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

sns.set_theme(style="whitegrid")


def titulo(texto):
    print(f"\n{texto}\n")


def mostrar_tabla(df, decimales=4):
    tabla = df.copy()

    for columna in tabla.select_dtypes(include=np.number).columns:
        tabla[columna] = tabla[columna].round(decimales)

    print(tabulate(
        tabla.reset_index(),
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    ))


def guardar_csv(df, nombre):
    df.to_csv(OUTPUT_DIR / nombre, index=False)


def guardar_grafico(nombre):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / nombre,
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


# 1. CARGA DE DATOS

titulo("1. CARGA DE DATOS")

if not DATA_PATH.exists():
    print("ERROR: No se encontró el archivo:")
    print(DATA_PATH)
    print("\nVerifique que Churn_Modelling.csv esté dentro de ProyectoM4.")
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)

filas_originales = len(df)
variables_originales = df.shape[1]

print(f"Registros: {filas_originales}")
print(f"Variables: {variables_originales}")
print("\nDataset cargado correctamente.")


# 2. VALIDACIÓN DEL DATASET

titulo("2. VALIDACIÓN DEL DATASET")

variables_validacion = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]

nulos = pd.DataFrame({
    "Variable": variables_validacion,
    "Valores_Nulos": [
        df[col].isna().sum()
        for col in variables_validacion
    ]
})

mostrar_tabla(nulos)

duplicados = df.duplicated().sum()

print(f"\nFilas duplicadas: {duplicados}")

if duplicados > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print(f"Registros después de validación: {len(df)}")


# 3. PREPARACIÓN DE DATOS

titulo("3. PREPARACIÓN DE DATOS")

continuous_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
]

binary_features = [
    "HasCrCard",
    "IsActiveMember"
]

categorical_features = [
    "Geography",
    "Gender"
]

numeric_features = continuous_features + binary_features
features = numeric_features + categorical_features

target = "Exited"

columnas_requeridas = features + [target]
faltantes = [col for col in columnas_requeridas if col not in df.columns]

if faltantes:
    raise ValueError(
        f"Faltan columnas requeridas en el dataset: {faltantes}"
    )

print("Variables utilizadas para clustering:")

for variable in features:
    print(f"- {variable}")

print("\nVariables binarias mantenidas en escala 0-1:")
for variable in binary_features:
    print(f"- {variable}")

print("\nVariable objetivo utilizada solamente para análisis:")
print("- Exited")

print("\nLas variables RowNumber, CustomerId y Surname no se utilizan.")
print("Exited tampoco se utiliza para construir los clusters.")


# 4. PREPROCESAMIENTO

titulo("4. PREPROCESAMIENTO")

X_continuous = df[continuous_features].astype(float).copy()
X_binary = df[binary_features].astype(float).copy()
X_categorical = df[categorical_features].copy()

scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X_continuous)

encoder = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

try:
    X_categorical_encoded = encoder.fit_transform(X_categorical)
except TypeError:
    encoder = OneHotEncoder(
        drop="first",
        sparse=False,
        handle_unknown="ignore"
    )
    X_categorical_encoded = encoder.fit_transform(X_categorical)

categorical_names = encoder.get_feature_names_out(
    categorical_features
)

categorical_names = [
    nombre.replace(" ", "_")
    for nombre in categorical_names
]

X_processed_raw = np.hstack([
    X_continuous_scaled,
    X_binary.to_numpy(),
    X_categorical_encoded
])

feature_names_clean = (
    continuous_features
    + binary_features
    + categorical_names
)

# Estandarización final para colocar todas las variables
# utilizadas por el clustering en una escala comparable.
final_scaler = StandardScaler()

X_processed = final_scaler.fit_transform(
    X_processed_raw
)

X_processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names_clean,
    index=df.index
)

print(f"Variables originales: {len(features)}")
print(f"Variables después del encoding: {len(feature_names_clean)}")

print("\nVariables finales del modelo:")

for variable in feature_names_clean:
    print(f"- {variable}")

guardar_csv(
    X_processed_df.reset_index(drop=True),
    "datos_preprocesados.csv"
)


# 5. SELECCIÓN DEL NÚMERO DE CLUSTERS

titulo("5. SELECCIÓN DEL NÚMERO DE CLUSTERS")

# Muestra común para la evaluación de Silhouette
SILHOUETTE_SAMPLE_SIZE = min(
    3000,
    len(df)
)

rng_silhouette = np.random.default_rng(
    RANDOM_STATE
)

silhouette_indices = rng_silhouette.choice(
    len(df),
    size=SILHOUETTE_SAMPLE_SIZE,
    replace=False
)

k_values = list(range(2, 9))

evaluacion_kmeans = []

for k in k_values:

    modelo = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = modelo.fit_predict(X_processed)

    score = silhouette_score(
        X_processed[silhouette_indices],
        labels[silhouette_indices]
    )

    evaluacion_kmeans.append({
        "K": k,
        "Inertia": modelo.inertia_,
        "Silhouette": score
    })

evaluacion_kmeans = pd.DataFrame(evaluacion_kmeans)

mostrar_tabla(evaluacion_kmeans)

guardar_csv(
    evaluacion_kmeans,
    "evaluacion_kmeans.csv"
)

best_k = int(
    evaluacion_kmeans.loc[
        evaluacion_kmeans["Silhouette"].idxmax(),
        "K"
    ]
)

best_silhouette = evaluacion_kmeans["Silhouette"].max()

print(f"\nMejor K según Silhouette Score: {best_k}")
print(f"Silhouette Score: {best_silhouette:.4f}")


# Gráfico del codo

plt.figure(figsize=(8, 5))

plt.plot(
    evaluacion_kmeans["K"],
    evaluacion_kmeans["Inertia"],
    marker="o"
)

plt.xlabel("Número de clusters (K)")
plt.ylabel("Inertia")
plt.title("Método del codo - K-Means")
plt.xticks(k_values)

guardar_grafico("metodo_codo_kmeans.png")


# Gráfico silhouette

plt.figure(figsize=(8, 5))

plt.plot(
    evaluacion_kmeans["K"],
    evaluacion_kmeans["Silhouette"],
    marker="o"
)

plt.xlabel("Número de clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score por número de clusters")
plt.xticks(k_values)

guardar_grafico("silhouette_kmeans.png")


# 6. K-MEANS

titulo("6. K-MEANS")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=20
)

kmeans_labels = kmeans.fit_predict(X_processed)

df["KMeans_Cluster"] = kmeans_labels

print(f"Clusters seleccionados: {best_k}")
print(f"Silhouette Score: {best_silhouette:.4f}")

cantidad_clusters = (
    df["KMeans_Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cantidad_clusters.columns = [
    "Cluster",
    "Clientes"
]

mostrar_tabla(cantidad_clusters, 0)

guardar_csv(
    cantidad_clusters,
    "cantidad_clientes_kmeans.csv"
)


# Centroides

centroides = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=feature_names_clean
)

centroides.insert(
    0,
    "Cluster",
    range(best_k)
)

mostrar_tabla(centroides)

guardar_csv(
    centroides,
    "centroides_kmeans.csv"
)


# 7. PCA

titulo("7. PCA")

pca = PCA()

X_pca = pca.fit_transform(X_processed)

varianza_pca = pd.DataFrame({
    "Componente": [
        f"PC{i+1}"
        for i in range(len(pca.explained_variance_ratio_))
    ],
    "Varianza_Explicada":
        pca.explained_variance_ratio_,
    "Varianza_Acumulada":
        np.cumsum(pca.explained_variance_ratio_)
})

mostrar_tabla(varianza_pca, 4)

guardar_csv(
    varianza_pca,
    "varianza_explicada_pca.csv"
)

print(
    f"\nVarianza explicada por PC1 + PC2: "
    f"{pca.explained_variance_ratio_[:2].sum():.4f}"
)

print(
    "\nNota: PCA se utiliza principalmente para "
    "reducir dimensionalidad y facilitar la visualización."
)

print(
    f"Varianza explicada por PC1 + PC2 + PC3: "
    f"{pca.explained_variance_ratio_[:3].sum():.4f}"
)

print(
    "La representación mediante PC1 y PC2 "
    "no contiene toda la información del espacio original."
)


# PCA 2D

pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_processed)

pca_df = pd.DataFrame({
    "PC1": X_pca_2d[:, 0],
    "PC2": X_pca_2d[:, 1],
    "Cluster": kmeans_labels
})

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2",
    s=35,
    alpha=0.65
)

plt.title("K-Means visualizado mediante PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")

guardar_grafico("kmeans_pca_2d.png")


# PCA 3D

pca_3d = PCA(n_components=3)

X_pca_3d = pca_3d.fit_transform(X_processed)

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=kmeans_labels,
    cmap="viridis",
    s=20,
    alpha=0.6
)

ax.set_title("K-Means visualizado mediante PCA 3D")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

guardar_grafico("kmeans_pca_3d.png")


# 8. PERFIL DE CLIENTES POR CLUSTER

titulo("8. PERFIL DE CLIENTES POR CLUSTER")

perfil_numerico = (
    df.groupby("KMeans_Cluster")[numeric_features]
    .mean()
)

print("Características numéricas:")

mostrar_tabla(perfil_numerico, 2)

guardar_csv(
    perfil_numerico.reset_index(),
    "perfil_numerico_kmeans.csv"
)


# Geografía

perfil_geografia = pd.crosstab(
    df["KMeans_Cluster"],
    df["Geography"],
    normalize="index"
)

print("\nDistribución geográfica:")

mostrar_tabla(perfil_geografia, 3)

guardar_csv(
    perfil_geografia.reset_index(),
    "perfil_geografia_kmeans.csv"
)


# Género

perfil_genero = pd.crosstab(
    df["KMeans_Cluster"],
    df["Gender"],
    normalize="index"
)

print("\nDistribución por género:")

mostrar_tabla(perfil_genero, 3)

guardar_csv(
    perfil_genero.reset_index(),
    "perfil_genero_kmeans.csv"
)


# 9. RELACIÓN ENTRE CLUSTERS Y CHURN

titulo("9. RELACIÓN ENTRE CLUSTERS Y CHURN")

churn_cluster = (
    df.groupby("KMeans_Cluster")
    .agg(
        Clientes=("Exited", "count"),
        Clientes_Churn=("Exited", "sum")
    )
)

churn_cluster["Tasa_Churn"] = (
    churn_cluster["Clientes_Churn"]
    / churn_cluster["Clientes"]
    * 100
)

mostrar_tabla(churn_cluster, 2)

guardar_csv(
    churn_cluster.reset_index(),
    "churn_por_cluster.csv"
)


# Gráfico churn

plt.figure(figsize=(8, 5))

sns.barplot(
    data=churn_cluster.reset_index(),
    x="KMeans_Cluster",
    y="Tasa_Churn"
)

plt.title("Tasa de churn por cluster - K-Means")
plt.xlabel("Cluster")
plt.ylabel("Tasa de churn (%)")

guardar_grafico("tasa_churn_kmeans.png")


# 10. RESUMEN DE SEGMENTOS

titulo("10. RESUMEN DE SEGMENTOS")

resumen_segmentos = (
    df.groupby("KMeans_Cluster")
    .agg(
        Clientes=("Exited", "count"),
        Edad_Promedio=("Age", "mean"),
        Balance_Promedio=("Balance", "mean"),
        CreditScore_Promedio=("CreditScore", "mean"),
        Productos_Promedio=("NumOfProducts", "mean"),
        Salario_Promedio=("EstimatedSalary", "mean"),
        Miembros_Activos_Pct=("IsActiveMember", "mean"),
        Tarjeta_Credito_Pct=("HasCrCard", "mean"),
        Churn_Pct=("Exited", "mean")
    )
    .reset_index()
)

resumen_segmentos["Miembros_Activos_Pct"] *= 100
resumen_segmentos["Tarjeta_Credito_Pct"] *= 100
resumen_segmentos["Churn_Pct"] *= 100

mostrar_tabla(resumen_segmentos, 2)

guardar_csv(
    resumen_segmentos,
    "resumen_segmentos.csv"
)


# 11. ANÁLISIS DE LOS SEGMENTOS

titulo("11. ANÁLISIS DE LOS SEGMENTOS")

cluster_mayor_churn = int(
    resumen_segmentos.loc[
        resumen_segmentos["Churn_Pct"].idxmax(),
        "KMeans_Cluster"
    ]
)

cluster_menor_churn = int(
    resumen_segmentos.loc[
        resumen_segmentos["Churn_Pct"].idxmin(),
        "KMeans_Cluster"
    ]
)

mayor_churn = resumen_segmentos["Churn_Pct"].max()
menor_churn = resumen_segmentos["Churn_Pct"].min()

print(
    f"El Cluster {cluster_mayor_churn} presenta "
    f"la mayor tasa de churn: {mayor_churn:.2f}%."
)

print(
    f"El Cluster {cluster_menor_churn} presenta "
    f"la menor tasa de churn: {menor_churn:.2f}%."
)

print("\nInterpretación de segmentos:")

for _, fila in resumen_segmentos.iterrows():

    cluster = int(fila["KMeans_Cluster"])

    print(f"\nCluster {cluster}:")
    print(f"- Clientes: {int(fila['Clientes'])}")
    print(f"- Edad promedio: {fila['Edad_Promedio']:.2f}")
    print(
        f"- Balance promedio: "
        f"{fila['Balance_Promedio']:,.2f}"
    )
    print(
        f"- CreditScore promedio: "
        f"{fila['CreditScore_Promedio']:.2f}"
    )
    print(
        f"- Productos promedio: "
        f"{fila['Productos_Promedio']:.2f}"
    )
    print(
        f"- Miembros activos: "
        f"{fila['Miembros_Activos_Pct']:.2f}%"
    )
    print(
        f"- Tarjeta de crédito: "
        f"{fila['Tarjeta_Credito_Pct']:.2f}%"
    )
    print(
        f"- Churn: "
        f"{fila['Churn_Pct']:.2f}%"
    )


# 12. DBSCAN

titulo("12. DBSCAN")

print("Evaluando diferentes combinaciones de eps y min_samples...")

# Se reduce la dimensionalidad antes de aplicar DBSCAN para trabajar en un espacio más estable.

pca_dbscan_completo = PCA()
X_dbscan_completo = pca_dbscan_completo.fit_transform(X_processed)

varianza_acumulada_dbscan = np.cumsum(
    pca_dbscan_completo.explained_variance_ratio_
)

n_componentes_dbscan = np.argmax(
    varianza_acumulada_dbscan >= 0.90
) + 1

pca_dbscan = PCA(
    n_components=n_componentes_dbscan
)

X_dbscan = pca_dbscan.fit_transform(X_processed)

print(f"Dimensiones originales: {X_processed.shape[1]}")
print(
    f"Dimensiones utilizadas por DBSCAN: "
    f"{X_dbscan.shape[1]}"
)
print(
    f"Varianza acumulada utilizada por DBSCAN: "
    f"{pca_dbscan.explained_variance_ratio_.sum():.4f}"
)


# Se evalúa una combinación amplia pero manejable de parámetros.

eps_values = [
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5
]

min_samples_values = [
    5,
    10,
    15,
    20
]

evaluacion_dbscan = []

for eps in eps_values:

    for min_samples in min_samples_values:

        modelo_dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )

        labels = modelo_dbscan.fit_predict(X_dbscan)

        mascara_no_ruido = labels != -1

        n_clusters = len(
            set(labels[mascara_no_ruido])
        )

        n_ruido = int(
            np.sum(labels == -1)
        )

        silhouette = np.nan

        # El Silhouette se calcula sobre una muestra para reducir el costo computacional.

        if (
            n_clusters >= 2
            and mascara_no_ruido.sum() >= 20
        ):

            indices_validos = np.where(
                mascara_no_ruido
            )[0]

            indices_validos = silhouette_indices[
                labels[silhouette_indices] != -1
            ]

            labels_validos = labels[indices_validos]

            if len(np.unique(labels_validos)) >= 2:

                silhouette = silhouette_score(
                    X_dbscan[indices_validos],
                    labels_validos
                )

        porcentaje_ruido = (
            n_ruido / len(df) * 100
        )

        evaluacion_dbscan.append({
            "eps": eps,
            "min_samples": min_samples,
            "Clusters": n_clusters,
            "Clientes_Ruido": n_ruido,
            "Porcentaje_Ruido": porcentaje_ruido,
            "Silhouette": silhouette
        })

evaluacion_dbscan = pd.DataFrame(
    evaluacion_dbscan
)

mostrar_tabla(
    evaluacion_dbscan,
    4
)

guardar_csv(
    evaluacion_dbscan,
    "evaluacion_dbscan.csv"
)


# Selección de DBSCAN

validos_dbscan = evaluacion_dbscan[
    evaluacion_dbscan["Silhouette"].notna()
].copy()

if len(validos_dbscan) == 0:

    print(
        "\nNo se encontró una configuración de DBSCAN "
        "con al menos dos clusters válidos."
    )

    best_eps = eps_values[0]
    best_min_samples = min_samples_values[0]

else:

    # Se prioriza Silhouette sin aceptar configuraciones con una proporción excesiva de ruido.

    validos_preferidos = validos_dbscan[
        validos_dbscan["Porcentaje_Ruido"] < 50
    ]

    print(
        "\nCriterio de selección DBSCAN:"
        "\n- Se consideran configuraciones con al menos 2 clusters."
        "\n- Se priorizan configuraciones con menos del 50% de ruido."
        "\n- Entre ellas, se selecciona la de mayor Silhouette Score."
    )

    if len(validos_preferidos) > 0:

        mejor_dbscan = validos_preferidos.loc[
            validos_preferidos["Silhouette"].idxmax()
        ]

    else:

        mejor_dbscan = validos_dbscan.loc[
            validos_dbscan["Silhouette"].idxmax()
        ]

    best_eps = mejor_dbscan["eps"]
    best_min_samples = int(
        mejor_dbscan["min_samples"]
    )


# Modelo final DBSCAN

dbscan = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples,
    n_jobs=-1
)

dbscan_labels = dbscan.fit_predict(X_dbscan)

df["DBSCAN_Cluster"] = dbscan_labels

mascara_no_ruido = dbscan_labels != -1

n_clusters_dbscan = len(
    set(dbscan_labels[mascara_no_ruido])
)

n_ruido = int(
    np.sum(dbscan_labels == -1)
)

silhouette_dbscan = np.nan

if (
    n_clusters_dbscan >= 2
    and mascara_no_ruido.sum() >= 20
):

    indices_validos = np.where(
        mascara_no_ruido
    )[0]

    indices_validos = silhouette_indices[
        dbscan_labels[silhouette_indices] != -1
    ]

    labels_validos = dbscan_labels[
        indices_validos
    ]

    if len(np.unique(labels_validos)) >= 2:

        silhouette_dbscan = silhouette_score(
            X_dbscan[indices_validos],
            labels_validos
        )

print("\nMejor configuración DBSCAN:")
print(f"eps = {best_eps}")
print(f"min_samples = {best_min_samples}")
print(f"Número de clusters encontrados: {n_clusters_dbscan}")
print(f"Clientes considerados ruido: {n_ruido}")

if pd.notna(silhouette_dbscan):
    print(
        f"Silhouette Score DBSCAN: "
        f"{silhouette_dbscan:.4f}"
    )
else:
    print("Silhouette Score DBSCAN: No disponible")


# 13. VISUALIZACIÓN DBSCAN

titulo("13. VISUALIZACIÓN DBSCAN")

pca_visual = PCA(n_components=2)

X_visual = pca_visual.fit_transform(X_processed)

dbscan_visual = pd.DataFrame({
    "PC1": X_visual[:, 0],
    "PC2": X_visual[:, 1],
    "Cluster": dbscan_labels
})

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=dbscan_visual,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab20",
    s=35,
    alpha=0.65
)

plt.title("DBSCAN visualizado mediante PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")

guardar_grafico("dbscan_pca_2d.png")


# 14. COMPARACIÓN K-MEANS VS DBSCAN

titulo("14. COMPARACIÓN K-MEANS VS DBSCAN")

porcentaje_ruido_dbscan = (
    n_ruido / len(df) * 100
)

comparacion_clustering = pd.DataFrame({
    "Metodo": [
        "K-Means",
        "DBSCAN"
    ],
    "Clusters": [
        best_k,
        n_clusters_dbscan
    ],
    "Silhouette_Score": [
        best_silhouette,
        silhouette_dbscan
    ],
    "Clientes_Ruido": [
        0,
        n_ruido
    ],
    "Porcentaje_Ruido": [
        0,
        porcentaje_ruido_dbscan
    ]
})

mostrar_tabla(
    comparacion_clustering,
    4
)

guardar_csv(
    comparacion_clustering,
    "comparacion_kmeans_dbscan.csv"
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparacion_clustering,
    x="Metodo",
    y="Silhouette_Score"
)

plt.title("Comparación del Silhouette Score")
plt.xlabel("Método de clustering")
plt.ylabel("Silhouette Score")

guardar_grafico("comparacion_silhouette.png")

# 15. t-SNE

titulo("15. t-SNE")

TSNE_SAMPLE_SIZE = min(
    3000,
    len(df)
)

rng = np.random.default_rng(
    RANDOM_STATE
)

if len(df) > TSNE_SAMPLE_SIZE:

    tsne_indices = rng.choice(
        len(df),
        size=TSNE_SAMPLE_SIZE,
        replace=False
    )

else:

    tsne_indices = np.arange(len(df))

print(
    f"Registros utilizados para t-SNE: "
    f"{len(tsne_indices)}"
)

X_tsne_input = X_processed[
    tsne_indices
]

clusters_tsne = kmeans_labels[
    tsne_indices
]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=RANDOM_STATE,
    max_iter=1000
)

X_tsne = tsne.fit_transform(
    X_tsne_input
)

resultado_tsne = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "KMeans_Cluster": clusters_tsne
})

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=resultado_tsne,
    x="TSNE1",
    y="TSNE2",
    hue="KMeans_Cluster",
    palette="Set2",
    s=35,
    alpha=0.65
)

plt.title("Visualización de clusters mediante t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

guardar_grafico("tsne_clusters.png")

guardar_csv(
    resultado_tsne,
    "resultado_tsne.csv"
)


# 16. RELACIÓN ENTRE CLUSTERS Y CHURN

titulo("16. RELACIÓN ENTRE CLUSTERS Y CHURN")

churn_kmeans = (
    df.groupby("KMeans_Cluster")
    .agg(
        Clientes=("Exited", "count"),
        Clientes_Churn=("Exited", "sum")
    )
)

churn_kmeans["Tasa_Churn"] = (
    churn_kmeans["Clientes_Churn"]
    / churn_kmeans["Clientes"]
    * 100
)

mostrar_tabla(
    churn_kmeans,
    2
)

guardar_csv(
    churn_kmeans.reset_index(),
    "tasa_churn_kmeans.csv"
)


# 17. FEATURE DERIVADA DEL CLUSTERING

titulo("17. FEATURE DERIVADA DEL CLUSTERING")

segment_mapping = {
    cluster: f"Segmento_{cluster}"
    for cluster in sorted(
        df["KMeans_Cluster"].unique()
    )
}

df["Customer_Segment"] = (
    df["KMeans_Cluster"]
    .map(segment_mapping)
)

print(
    "Se creó la variable derivada: "
    "Customer_Segment"
)


# 18. CHURN POR CLUSTER DBSCAN

titulo("18. CHURN POR CLUSTER DBSCAN")

churn_dbscan = (
    df.groupby("DBSCAN_Cluster")
    .agg(
        Clientes=("Exited", "count"),
        Clientes_Churn=("Exited", "sum")
    )
)

churn_dbscan["Tasa_Churn"] = (
    churn_dbscan["Clientes_Churn"]
    / churn_dbscan["Clientes"]
    * 100
)

mostrar_tabla(
    churn_dbscan,
    2
)

guardar_csv(
    churn_dbscan.reset_index(),
    "churn_por_cluster_dbscan.csv"
)


# 19. DATASET SEGMENTADO

titulo("19. DATASET SEGMENTADO")

guardar_csv(
    df,
    "dataset_segmentado.csv"
)

clientes_segmentacion = df[
    [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "Exited",
        "KMeans_Cluster",
        "DBSCAN_Cluster",
        "Customer_Segment"
    ]
].copy()

guardar_csv(
    clientes_segmentacion,
    "clientes_con_segmentacion.csv"
)


# 20. RESUMEN GENERAL

titulo("20. RESUMEN GENERAL")

resumen_general = pd.DataFrame({
    "Indicador": [
        "Registros",
        "Variables originales",
        "Variables utilizadas",
        "Mejor K-Means",
        "Silhouette K-Means",
        "Clusters DBSCAN",
        "Clientes ruido DBSCAN",
        "Porcentaje ruido DBSCAN",
        "Silhouette DBSCAN",
        "Cluster mayor churn",
        "Mayor tasa churn",
        "Cluster menor churn",
        "Menor tasa churn"
    ],
    "Valor": [
        len(df),
        variables_originales,
        len(features),
        best_k,
        best_silhouette,
        n_clusters_dbscan,
        n_ruido,
        porcentaje_ruido_dbscan,
        silhouette_dbscan,
        cluster_mayor_churn,
        mayor_churn,
        cluster_menor_churn,
        menor_churn
    ]
})

mostrar_tabla(
    resumen_general,
    4
)

guardar_csv(
    resumen_general,
    "resumen_general.csv"
)


# 21. EXPORTACIÓN FINAL

titulo("21. EXPORTACIÓN FINAL")

print("Todos los resultados fueron guardados en:")
print(OUTPUT_DIR)

print("\nArchivos principales creados:")

archivos = sorted(
    [
        archivo.name
        for archivo in OUTPUT_DIR.iterdir()
        if archivo.is_file()
    ]
)

for archivo in archivos:
    print(f"- {archivo}")


# RESUMEN FINAL

titulo("RESUMEN FINAL")

print(
    f"Mejor número de clusters K-Means: {best_k}"
)

print(
    f"Silhouette K-Means: "
    f"{best_silhouette:.4f}"
)

print(
    f"Clusters encontrados por DBSCAN: "
    f"{n_clusters_dbscan}"
)

print(
    f"Clientes identificados como ruido por DBSCAN: "
    f"{n_ruido}"
)

if pd.notna(silhouette_dbscan):

    print(
        f"Silhouette DBSCAN: "
        f"{silhouette_dbscan:.4f}"
    )

else:

    print(
        "Silhouette DBSCAN: No disponible"
    )

print(
    f"\nSegmento con mayor churn: "
    f"Cluster {cluster_mayor_churn} "
    f"({mayor_churn:.2f}%)"
)

print(
    f"Segmento con menor churn: "
    f"Cluster {cluster_menor_churn} "
    f"({menor_churn:.2f}%)"
)

print("\nAvance 3 finalizado correctamente.")

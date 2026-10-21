# Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tabulate import tabulate

# Scikit-learn
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# Modelos
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Configuracion
sns.set_theme(style="whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"

OUTPUT_DIR = BASE_DIR / "resultados_avance_2"
OUTPUT_DIR.mkdir(exist_ok=True)


# Funcion para mostrar secciones
def mostrar_seccion(numero, titulo):
    print("\n" + "=" * 80)
    print(f"{numero}. {titulo}")
    print("=" * 80)


# Funcion para limpiar nombres de variables
def limpiar_nombre(nombre):

    nombre = str(nombre)

    nombre = nombre.replace("num__", "")
    nombre = nombre.replace("cat__", "")

    nombre = nombre.replace("Geography_", "Geography = ")
    nombre = nombre.replace("Gender_", "Gender = ")

    return nombre


# Funcion para limpiar todos los nombres
def limpiar_nombres(nombres):
    return [limpiar_nombre(nombre) for nombre in nombres]


# Funcion para crear tabla de metricas
def crear_metricas(nombre, y_real, y_pred, y_prob):

    return {
        "Modelo": nombre,
        "Accuracy": round(accuracy_score(y_real, y_pred), 4),
        "Precision": round(
            precision_score(y_real, y_pred, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_real, y_pred, zero_division=0), 4
        ),
        "F1-Score": round(
            f1_score(y_real, y_pred, zero_division=0), 4
        ),
        "ROC-AUC": round(
            roc_auc_score(y_real, y_prob), 4
        ),
        "PR-AUC": round(
            average_precision_score(y_real, y_prob), 4
        )
    }


# Funcion para mostrar metricas
def mostrar_metricas(metricas):

    tabla = pd.DataFrame([metricas])

    print(
        tabulate(
            tabla,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            floatfmt=".4f"
        )
    )


# 1. CARGA DE DATOS
mostrar_seccion(1, "CARGA DE DATOS")

df = pd.read_csv(DATA_PATH)

print(f"Registros: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print("\nDataset cargado correctamente.")


# 2. PREPARACION DE DATOS
mostrar_seccion(2, "PREPARACION DE DATOS")

columns_to_drop = [
    "RowNumber",
    "CustomerId",
    "Surname"
]

X = df.drop(columns=columns_to_drop + ["Exited"])
y = df["Exited"]

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("Variables predictoras:")
print(", ".join(X.columns))

print("\nVariables numericas:")
print(", ".join(numeric_features))

print("\nVariables categoricas:")
print(", ".join(categorical_features))

print("\nVariable objetivo:")
print("Exited")


# 3. DIVISION TRAIN / TEST
mostrar_seccion(3, "DIVISION TRAIN / TEST")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)} registros")
print(f"Test:  {len(X_test)} registros")

train_churn = y_train.value_counts().sort_index()
test_churn = y_test.value_counts().sort_index()
total_churn = y.value_counts().sort_index()

distribucion = pd.DataFrame({
    "Clase": ["No Churn", "Churn"],
    "Train": [
        train_churn.get(0, 0),
        train_churn.get(1, 0)
    ],
    "Test": [
        test_churn.get(0, 0),
        test_churn.get(1, 0)
    ],
    "Total": [
        total_churn.get(0, 0),
        total_churn.get(1, 0)
    ]
})

print("\nDistribucion de churn:")
print(
    tabulate(
        distribucion,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)


# 4. PREPROCESAMIENTO
mostrar_seccion(4, "PREPROCESAMIENTO")

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
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)

preprocessor.fit(X_train)

feature_names = preprocessor.get_feature_names_out()
feature_names_clean = limpiar_nombres(feature_names)

print("Preprocesamiento completado.")

print(f"\nVariables originales: {X.shape[1]}")
print(f"Variables despues del encoding: {len(feature_names_clean)}")

print("\nVariables finales del modelo:")

variables_finales = pd.DataFrame({
    "Variable": feature_names_clean
})

print(
    tabulate(
        variables_finales,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)


# 5. VALIDACION CRUZADA
mostrar_seccion(5, "VALIDACION CRUZADA")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Metodo: StratifiedKFold")
print("Numero de folds: 5")
print("La proporcion de churn se mantiene en cada fold.")


# 6. MODELO REGRESION LOGISTICA
mostrar_seccion(6, "REGRESION LOGISTICA BASELINE")

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

metrics_logistic = crear_metricas(
    "Regresion Logistica",
    y_test,
    y_pred_logistic,
    y_prob_logistic
)

mostrar_metricas(metrics_logistic)


# 7. RANDOM FOREST
mostrar_seccion(7, "RANDOM FOREST")

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)
y_prob_rf = random_forest.predict_proba(X_test)[:, 1]

metrics_rf = crear_metricas(
    "Random Forest",
    y_test,
    y_pred_rf,
    y_prob_rf
)

mostrar_metricas(metrics_rf)


# 8. XGBOOST BASE
mostrar_seccion(8, "XGBOOST BASE")

xgb_base = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.10,
                subsample=0.90,
                colsample_bytree=0.90,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

xgb_base.fit(X_train, y_train)

y_pred_xgb = xgb_base.predict(X_test)
y_prob_xgb = xgb_base.predict_proba(X_test)[:, 1]

metrics_xgb = crear_metricas(
    "XGBoost",
    y_test,
    y_pred_xgb,
    y_prob_xgb
)

mostrar_metricas(metrics_xgb)


# 9. GRID SEARCH XGBOOST
mostrar_seccion(9, "OPTIMIZACION DE XGBOOST CON GRID SEARCH")

print("Buscando la mejor combinacion de hiperparametros...")

param_grid = {
    "classifier__learning_rate": [
        0.05,
        0.10,
        0.20
    ],
    "classifier__max_depth": [
        3,
        4,
        5
    ],
    "classifier__n_estimators": [
        200,
        300,
        400
    ]
}

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_

print("\nMejores parametros encontrados:")

best_params_df = pd.DataFrame({
    "Parametro": [
        "learning_rate",
        "max_depth",
        "n_estimators"
    ],
    "Valor": [
        grid_search.best_params_["classifier__learning_rate"],
        grid_search.best_params_["classifier__max_depth"],
        grid_search.best_params_["classifier__n_estimators"]
    ]
})

print(
    tabulate(
        best_params_df,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

print(
    f"\nMejor ROC-AUC promedio en Cross Validation: "
    f"{grid_search.best_score_:.4f}"
)

y_pred_xgb_opt = best_xgb.predict(X_test)
y_prob_xgb_opt = best_xgb.predict_proba(X_test)[:, 1]

metrics_xgb_opt = crear_metricas(
    "XGBoost Optimizado",
    y_test,
    y_pred_xgb_opt,
    y_prob_xgb_opt
)

print("\nResultado XGBoost Optimizado:")
mostrar_metricas(metrics_xgb_opt)


# 10. LIGHTGBM
mostrar_seccion(10, "LIGHTGBM")

lightgbm_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.90,
                colsample_bytree=0.90,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        )
    ]
)

lightgbm_model.fit(X_train, y_train)

y_pred_lgbm = lightgbm_model.predict(X_test)
y_prob_lgbm = lightgbm_model.predict_proba(X_test)[:, 1]

metrics_lgbm = crear_metricas(
    "LightGBM",
    y_test,
    y_pred_lgbm,
    y_prob_lgbm
)

mostrar_metricas(metrics_lgbm)


# 11. CATBOOST
mostrar_seccion(11, "CATBOOST")

catboost_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False,
                thread_count=-1
            )
        )
    ]
)

catboost_model.fit(X_train, y_train)

y_pred_cat = catboost_model.predict(X_test)
y_pred_cat = np.asarray(y_pred_cat).astype(int).ravel()

y_prob_cat = catboost_model.predict_proba(X_test)[:, 1]

metrics_cat = crear_metricas(
    "CatBoost",
    y_test,
    y_pred_cat,
    y_prob_cat
)

mostrar_metricas(metrics_cat)


# 12. COMPARACION DE MODELOS
mostrar_seccion(12, "COMPARACION DE MODELOS")

model_results = [
    metrics_rf,
    metrics_xgb,
    metrics_xgb_opt,
    metrics_lgbm,
    metrics_cat
]

comparison_df = pd.DataFrame(model_results)

comparison_df = comparison_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print(
    tabulate(
        comparison_df,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        floatfmt=".4f"
    )
)


# 13. COMPARACION CON REGRESION LOGISTICA
mostrar_seccion(13, "COMPARACION CON REGRESION LOGISTICA BASELINE")

comparison_with_baseline = pd.concat(
    [
        comparison_df,
        pd.DataFrame([metrics_logistic])
    ],
    ignore_index=True
)

comparison_with_baseline = comparison_with_baseline.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print(
    tabulate(
        comparison_with_baseline,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        floatfmt=".4f"
    )
)


# 14. CROSS VALIDATION
mostrar_seccion(14, "RESULTADOS DE CROSS VALIDATION")

models_for_cv = {
    "Regresion Logistica": logistic_model,
    "Random Forest": random_forest,
    "XGBoost": xgb_base,
    "XGBoost Optimizado": best_xgb,
    "LightGBM": lightgbm_model,
    "CatBoost": catboost_model
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

cv_results = []

for name, model in models_for_cv.items():

    print(f"Procesando {name}...")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({
        "Modelo": name,
        "Accuracy CV": scores["test_accuracy"].mean(),
        "Precision CV": scores["test_precision"].mean(),
        "Recall CV": scores["test_recall"].mean(),
        "F1 CV": scores["test_f1"].mean(),
        "ROC-AUC CV": scores["test_roc_auc"].mean(),
        "PR-AUC CV": scores["test_pr_auc"].mean()
    })

cv_results_df = pd.DataFrame(cv_results)

for column in cv_results_df.columns:
    if column != "Modelo":
        cv_results_df[column] = cv_results_df[column].round(4)

print(
    tabulate(
        cv_results_df,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        floatfmt=".4f"
    )
)


# 15. FEATURE IMPORTANCE
mostrar_seccion(15, "FEATURE IMPORTANCE")

fitted_models = {
    "Random Forest": random_forest,
    "XGBoost": xgb_base,
    "XGBoost Optimizado": best_xgb,
    "LightGBM": lightgbm_model,
    "CatBoost": catboost_model
}

feature_importance_results = {}


for model_name, model in fitted_models.items():

    classifier = model.named_steps["classifier"]

    if model_name == "LightGBM":

        importance_values = classifier.booster_.feature_importance(
            importance_type="gain"
        )

    elif model_name == "CatBoost":

        importance_values = classifier.get_feature_importance()

    else:

        importance_values = classifier.feature_importances_

    importance_values = np.asarray(
        importance_values,
        dtype=float
    )

    total_importance = importance_values.sum()

    if total_importance > 0:
        importance_percentage = (
            importance_values / total_importance
        ) * 100
    else:
        importance_percentage = importance_values

    importance_df = pd.DataFrame({
        "Variable": feature_names_clean,
        "Importancia (%)": importance_percentage
    })

    importance_df = importance_df.sort_values(
        by="Importancia (%)",
        ascending=False
    ).reset_index(drop=True)

    importance_df["Importancia (%)"] = importance_df[
        "Importancia (%)"
    ].round(4)

    feature_importance_results[model_name] = importance_df

    print(f"\nTop 10 variables - {model_name}:")

    print(
        tabulate(
            importance_df.head(10),
            headers="keys",
            tablefmt="grid",
            showindex=False,
            floatfmt=".4f"
        )
    )


# 16. GRAFICOS DE FEATURE IMPORTANCE
mostrar_seccion(16, "GRAFICOS DE FEATURE IMPORTANCE")

for model_name, importance_df in feature_importance_results.items():

    top_10 = importance_df.head(10).sort_values(
        by="Importancia (%)"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_10["Variable"],
        top_10["Importancia (%)"]
    )

    plt.title(
        f"Top 10 Feature Importance - {model_name}"
    )

    plt.xlabel("Importancia (%)")
    plt.ylabel("Variable")

    plt.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    graph_path = (
        OUTPUT_DIR /
        f"feature_importance_{safe_name}.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# 17. STACKING
mostrar_seccion(17, "STACKING")

print(
    "Construyendo ensamble de XGBoost, LightGBM y CatBoost..."
)

stacking_estimators = [
    (
        "xgboost",
        best_xgb
    ),
    (
        "lightgbm",
        lightgbm_model
    ),
    (
        "catboost",
        catboost_model
    )
]

stacking_model = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)

y_pred_stack = stacking_model.predict(X_test)
y_prob_stack = stacking_model.predict_proba(X_test)[:, 1]

metrics_stack = crear_metricas(
    "Stacking",
    y_test,
    y_pred_stack,
    y_prob_stack
)

mostrar_metricas(metrics_stack)


# 18. CROSS VALIDATION STACKING
mostrar_seccion(18, "CROSS VALIDATION STACKING")

stacking_cv_scores = cross_validate(
    stacking_model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

stacking_cv_result = pd.DataFrame([{
    "Modelo": "Stacking",
    "Accuracy CV": stacking_cv_scores["test_accuracy"].mean(),
    "Precision CV": stacking_cv_scores["test_precision"].mean(),
    "Recall CV": stacking_cv_scores["test_recall"].mean(),
    "F1 CV": stacking_cv_scores["test_f1"].mean(),
    "ROC-AUC CV": stacking_cv_scores["test_roc_auc"].mean(),
    "PR-AUC CV": stacking_cv_scores["test_pr_auc"].mean()
}])

for column in stacking_cv_result.columns:
    if column != "Modelo":
        stacking_cv_result[column] = stacking_cv_result[column].round(4)

print(
    tabulate(
        stacking_cv_result,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        floatfmt=".4f"
    )
)


# 19. COMPARACION FINAL
mostrar_seccion(19, "COMPARACION FINAL DE MODELOS")

final_results = pd.concat(
    [
        pd.DataFrame(model_results),
        pd.DataFrame([metrics_stack]),
        pd.DataFrame([metrics_logistic])
    ],
    ignore_index=True
)

final_results = final_results.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print(
    tabulate(
        final_results,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        floatfmt=".4f"
    )
)


# 20. MEJOR MODELO
mostrar_seccion(20, "MEJOR MODELO")

best_model_row = final_results.iloc[0]

best_model_name = best_model_row["Modelo"]
best_model_auc = best_model_row["ROC-AUC"]

print(
    f"Mejor modelo segun ROC-AUC: {best_model_name}"
)

print(
    f"ROC-AUC: {best_model_auc:.4f}"
)


# Seleccionar predicciones del mejor modelo
predictions = {
    "Regresion Logistica": (
        y_pred_logistic,
        y_prob_logistic
    ),
    "Random Forest": (
        y_pred_rf,
        y_prob_rf
    ),
    "XGBoost": (
        y_pred_xgb,
        y_prob_xgb
    ),
    "XGBoost Optimizado": (
        y_pred_xgb_opt,
        y_prob_xgb_opt
    ),
    "LightGBM": (
        y_pred_lgbm,
        y_prob_lgbm
    ),
    "CatBoost": (
        y_pred_cat,
        y_prob_cat
    ),
    "Stacking": (
        y_pred_stack,
        y_prob_stack
    )
}

best_pred, best_prob = predictions[best_model_name]


# 21. MATRIZ DE CONFUSION
mostrar_seccion(21, "MATRIZ DE CONFUSION")

cm = confusion_matrix(
    y_test,
    best_pred
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Real No Churn",
        "Real Churn"
    ],
    columns=[
        "Predicho No Churn",
        "Predicho Churn"
    ]
)

print(
    tabulate(
        cm_df,
        headers="keys",
        tablefmt="grid"
    )
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Predicho No Churn",
        "Predicho Churn"
    ],
    yticklabels=[
        "Real No Churn",
        "Real Churn"
    ]
)

plt.title(
    f"Matriz de Confusion - {best_model_name}"
)

plt.xlabel("Prediccion")
plt.ylabel("Valor real")

plt.tight_layout()

cm_path = OUTPUT_DIR / "matriz_confusion_mejor_modelo.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# 22. CURVA ROC
mostrar_seccion(22, "CURVA ROC")

roc_models = {
    "Regresion Logistica": y_prob_logistic,
    "Random Forest": y_prob_rf,
    "XGBoost": y_prob_xgb,
    "XGBoost Optimizado": y_prob_xgb_opt,
    "LightGBM": y_prob_lgbm,
    "CatBoost": y_prob_cat,
    "Stacking": y_prob_stack
}

plt.figure(figsize=(10, 7))

for name, probabilities in roc_models.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Clasificador aleatorio"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparacion de Curvas ROC")

plt.legend()
plt.tight_layout()

roc_path = OUTPUT_DIR / "comparacion_curvas_roc.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# 23. CURVA PRECISION-RECALL
mostrar_seccion(23, "CURVA PRECISION-RECALL")

plt.figure(figsize=(10, 7))

for name, probabilities in roc_models.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (PR-AUC={pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Comparacion de Curvas Precision-Recall")

plt.legend()
plt.tight_layout()

pr_path = OUTPUT_DIR / "comparacion_curvas_pr.png"

plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# 24. EXPORTACION DE RESULTADOS
mostrar_seccion(24, "EXPORTACION DE RESULTADOS")


# Metricas de modelos
final_results.to_csv(
    OUTPUT_DIR / "comparacion_final.csv",
    index=False,
    encoding="utf-8-sig"
)

comparison_df.to_csv(
    OUTPUT_DIR / "resultados_modelos.csv",
    index=False,
    encoding="utf-8-sig"
)

cv_results_df.to_csv(
    OUTPUT_DIR / "resultados_cross_validation.csv",
    index=False,
    encoding="utf-8-sig"
)

stacking_cv_result.to_csv(
    OUTPUT_DIR / "resultados_cross_validation_stacking.csv",
    index=False,
    encoding="utf-8-sig"
)


# Parametros XGBoost
best_params_df.to_csv(
    OUTPUT_DIR / "mejores_parametros_xgboost.csv",
    index=False,
    encoding="utf-8-sig"
)


# Resultado Stacking
pd.DataFrame([metrics_stack]).to_csv(
    OUTPUT_DIR / "resultado_stacking.csv",
    index=False,
    encoding="utf-8-sig"
)


# Feature Importance
for model_name, importance_df in feature_importance_results.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    importance_df.to_csv(
        OUTPUT_DIR /
        f"feature_importance_{safe_name}.csv",
        index=False,
        encoding="utf-8-sig"
    )


# Matriz de confusion
cm_export = pd.DataFrame(
    cm,
    index=[
        "Real No Churn",
        "Real Churn"
    ],
    columns=[
        "Predicho No Churn",
        "Predicho Churn"
    ]
)

cm_export.to_csv(
    OUTPUT_DIR / "matriz_confusion_mejor_modelo.csv",
    encoding="utf-8-sig"
)


print(
    f"Resultados exportados correctamente en:"
)

print(OUTPUT_DIR)

print("\nArchivos principales creados:")

for file in sorted(OUTPUT_DIR.iterdir()):

    if file.is_file():
        print(f"- {file.name}")


# 25. RESUMEN FINAL
mostrar_seccion(25, "RESUMEN DEL AVANCE 2")

print(
    f"Mejor modelo segun ROC-AUC: {best_model_name}"
)

print(
    f"ROC-AUC: {best_model_row['ROC-AUC']:.4f}"
)

print(
    f"Recall: {best_model_row['Recall']:.4f}"
)

print(
    f"F1-Score: {best_model_row['F1-Score']:.4f}"
)

print(
    f"PR-AUC: {best_model_row['PR-AUC']:.4f}"
)

print("\nAvance 2 finalizado correctamente.")
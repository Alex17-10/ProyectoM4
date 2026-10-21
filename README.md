> 🌐 **Note:** The Spanish version of this README is available as a PDF. Some content (charts, diagrams, screenshots) is easier to follow in the PDF version.
>
> 🌐 **Nota:** La versión en español de este README está disponible en formato PDF. Algunos contenidos (gráficas, diagramas, capturas) se ven más claros en la versión PDF.

---

# Integrative Project: Bank Churn

## Stage 4: Documentation and Consolidation

## Project Description

FinanceGuard is a digital bank that has grown exponentially in recent years, reaching 10,000 active customers. However, management has detected a worrying increase in the customer attrition rate (churn), currently at 20% per year, resulting in multimillion-dollar losses.

First, the structure and quality of the data were understood; then classification models were built and compared to estimate churn; finally, unsupervised learning techniques were applied to segment customers and study how the segments relate to the target variable.

Stage 4 does not add a new model or aim to show an additional analysis. Its role is to consolidate what was already executed, explain the decisions made, present the relevant results, and leave a clear path to reproduce or continue the project. The documentation is limited to procedures and results supported by the available scripts, dataset and results.

## Problem Addressed

The variable `Exited` indicates whether a customer left the institution. The problem combines two complementary needs: identifying how well customers with and without churn can be distinguished using supervised models, and discovering groups of customers with similar characteristics through clustering. Segmentation does not replace prediction; it works as an additional perspective for interpreting the heterogeneity of the customer base.

## Objective

As a Junior Data Scientist, the mission is to develop a complete Machine Learning model that predicts which customers are most likely to leave the bank. This model will allow the retention team to implement personalized strategies to retain at-risk customers, with the goal of reducing churn to 15%.

## Scope of Stage 4

The documentation covers the full flow from the input dataset to the segmented dataset. It includes data preparation, train/test split, cross-validation, supervised models, XGBoost optimization, logistic regression interpretability, model comparison, K-Means, DBSCAN, PCA, t-SNE, the relationship between segments and churn, export of results, and reproducibility.

## Data and Preparation

The input file used was `Churn_Modelling.csv`. It contains 10,000 records and 14 variables. The validation done in Stage 3 confirmed that the analyzed variables have no null values and that there are no duplicate rows. The same dataset was used as the reference for the classification and clustering processes.

### Dataset Variables

| Variable | Type | Main use |
|---|---|---|
| RowNumber | Numeric | Technical identifier; excluded from modeling |
| CustomerId | Numeric | Customer identifier; excluded from modeling |
| Surname | Categorical | Nominal data; excluded from modeling |
| CreditScore | Numeric | Predictor / clustering |
| Geography | Categorical | Predictor / clustering |
| Gender | Categorical | Predictor / clustering |
| Age | Numeric | Predictor / clustering |
| Tenure | Numeric | Predictor / clustering |
| Balance | Numeric | Predictor / clustering |
| NumOfProducts | Numeric | Predictor / clustering |
| HasCrCard | Binary numeric | Predictor / clustering |
| IsActiveMember | Binary numeric | Predictor / clustering |
| EstimatedSalary | Numeric | Predictor / clustering |
| Exited | Binary | Target variable; not used to form clusters |

In the supervised models, `RowNumber`, `CustomerId` and `Surname` were removed and `Exited` was used as the target variable. Numeric variables were standardized with `StandardScaler` and categorical ones with `OneHotEncoder`, dropping the first category. For clustering, binary variables were kept on a 0–1 scale before the final standardization, and `OneHotEncoder` was also applied to `Geography` and `Gender`.

### Initial Churn Distribution

> **Figure (see PDF):** bar chart "Customer distribution by churn" across the 10,000 customers.

The dataset contains 7,963 customers without churn and 2,037 with churn, equivalent to 79.63% and 20.37% respectively. This distribution confirms that the churn class is a minority, so evaluation should not rely on Accuracy alone.

## Continuity of the Three Stages

The project should be understood as a single analytical flow. Each stage addressed a different part of the problem, and the results of one stage serve as the base for the next.

### Understanding and Baseline

The first stage focused on exploratory analysis, dataset quality, variable classification, data preparation, and building a Logistic Regression as a baseline model. Correlations, multicollinearity through VIF, coefficients, Odds Ratios and confidence intervals were also studied.

### Improving the Prediction

The second stage expanded the modeling with Random Forest, XGBoost, optimized XGBoost, LightGBM and CatBoost. It added stratified five-fold cross-validation, Grid Search for XGBoost, metric comparison, and a Stacking model made up of XGBoost, LightGBM and CatBoost with Logistic Regression as the meta-learner.

### Customer Segmentation

The third stage moved the analysis to unsupervised learning. The number of clusters for K-Means was evaluated using Inertia and Silhouette, three segments were built, each group was profiled, and their churn rates were compared. DBSCAN was then evaluated with different combinations of `eps` and `min_samples`. PCA and t-SNE were used as dimensionality reduction and visualization tools.

This sequence moves from a general question (what characterizes churn?) to two complementary perspectives: who can be classified as a higher-risk customer, and how customers group according to their characteristics.

## Exploratory Analysis and Baseline Model

The initial analysis established a quantitative base for understanding the problem. Dimensions, missing values, duplicates, descriptive statistics and categorical variables were reviewed. The dataset had no missing values in the variables used for clustering and no duplicates.

Classification preparation separated the predictor variables from `Exited` and discarded identifiers that did not provide a useful behavioral or financial characteristic for the model. The train/test split used 80% of the data for training and 20% for testing, with `random_state=42` and stratification to preserve the churn proportion.

| Set | Records | Proportion | Churn | No churn |
|---|---|---|---|---|
| Training | 8,000 | 80% | 1,630 | 6,370 |
| Test | 2,000 | 20% | 407 | 1,593 |
| Total | 10,000 | 100% | 2,037 | 7,963 |

### Baseline Logistic Regression

| Metric | Result |
|---|---|
| Accuracy | 0.8080 |
| Precision | 0.5891 |
| Recall | 0.1867 |
| F1-Score | 0.2836 |
| ROC-AUC | 0.7748 |

The baseline reached a ROC-AUC of 0.7748 but showed a Recall of 0.1867 for the churn class. This means that, at the threshold used, it identifies only a limited share of the customers who actually left. For this reason, Logistic Regression works as the reference point to assess whether later models improve discrimination and recovery of the positive class.

### Baseline Interpretability

| Variable | Odds Ratio | OR 95% CI | Reading |
|---|---|---|---|
| Age | 2.0928 | 1.9760–2.2219 | Positive association with churn odds |
| Geography = Germany | 2.2811 | 1.9671–2.6451 | Positive association with churn odds |
| Balance | 1.1741 | 1.0942–1.2597 | Positive association with churn odds |
| IsActiveMember | 0.5968 | 0.5605–0.6355 | Negative association with churn odds |
| Gender = Male | 0.5917 | 0.5252–0.6666 | Negative association with churn odds |
| CreditScore | 0.9175 | 0.8648–0.9735 | Negative association with churn odds |

The intervals show which estimates have statistical evidence consistent with an association different from an Odds Ratio of 1. These results describe associations within the model; they should not be interpreted as causal relationships.

In particular, `Age` and `Geography = Germany` have odds above 1, while `IsActiveMember` and `Gender = Male` have odds below 1. The interpretation should remain in terms of association and with respect to the other variables included in the model.

## Supervised Modeling, Optimization and Comparison

The second stage extended the baseline with tree-based and boosting models: Random Forest, XGBoost, optimized XGBoost, LightGBM and CatBoost. Evaluation used Accuracy, Precision, Recall, F1-Score, ROC-AUC and PR-AUC. Cross-validation was configured with five-fold `StratifiedKFold`, and ROC-AUC was used as the criterion for the XGBoost Grid Search.

### Comparative Discrimination Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.7748 | 0.4790 |
| Random Forest | 0.849 | 0.674 |
| XGBoost | 0.860 | 0.708 |
| Optimized XGBoost | 0.869 | 0.716 |
| LightGBM | 0.861 | 0.703 |
| CatBoost | 0.868 | 0.725 |
| Stacking | 0.870 | 0.724 |

The ROC-AUC and PR-AUC values come from the curves generated in Stage 2. The improvement over the baseline is consistent: boosting models reach higher discrimination ability, and Stacking obtains the highest ROC-AUC among the available results, at 0.870. CatBoost has the highest PR-AUC in the comparison, at 0.725.

### Optimized XGBoost

The hyperparameter search evaluated `learning_rate`, `max_depth` and `n_estimators`. The selected configuration was `learning_rate=0.05`, `max_depth=3` and `n_estimators=300`. The mean cross-validation ROC-AUC recorded for that configuration was 0.8657; on the test set it obtained Accuracy 0.8695, Precision 0.7874, Recall 0.4914, F1-Score 0.6051, ROC-AUC 0.8689 and PR-AUC 0.7201.

The result shows that optimization improved on base XGBoost on the test set, especially in the discrimination metrics. However, Recall remains below 0.50, so the ability to recover all churning customers remains a relevant issue.

### Feature Importance

> **Figure (see PDF):** "Top 10 Feature Importance - Optimized XGBoost" bar chart.

Variables are ranked by importance in the optimized XGBoost. Importance is relative to the model and does not imply causality.

In the optimized XGBoost, `NumOfProducts`, `Age` and `IsActiveMember` account for the largest relative importances. `Geography = Germany` also stands out compared to other variables. The correct reading is that these variables contributed more to this model's predictive process; it does not mean that they cause churn on their own.

### Final Evaluation and Stacking

Stacking combines the probabilistic predictions of XGBoost, LightGBM and CatBoost and uses a Logistic Regression as the meta-learner. The architecture uses internal five-fold cross-validation and `predict_proba` as input to the upper level.

> **Figure (see PDF):** confusion matrix of the Stacking model on the test set.

The matrix contains 1,540 true negatives, 53 false positives, 203 false negatives and 204 true positives. From these values, the model reaches Accuracy of 0.8720, Precision of 0.7938, Recall of 0.5012 and F1-Score of 0.6145. The ROC curve records an AUC of 0.870 and the Precision-Recall curve records a PR-AUC of 0.724.

The matrix shows an important characteristic: although the model improves notably over the baseline in recovering churn, there are still 203 churning customers classified as non-churn. Therefore, choosing an operating threshold would depend on the business objective and on the relative cost of false negatives and false positives.

### Evaluation Curves

The comparison of ROC and Precision-Recall curves was used during Stage 2 to evaluate model behavior at different thresholds. The documentation keeps the ROC-AUC and PR-AUC values in the comparison table, avoiding a multicolor figure in the final document. Reading both metrics together is especially relevant because the churn class represents roughly one fifth of the dataset.

## Unsupervised Segmentation

Stage 3 uses ten original variables for clustering: six continuous variables, two binary and two categorical. `Geography` and `Gender` were transformed with `OneHotEncoder`; continuous variables were standardized, and a final standardization was then applied to all variables entering the clustering. `Exited` was left out of cluster construction and used only afterward to analyze churn rates.

### K-Means Selection

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 97,252.4 | 0.1325 |
| 3 | 88,348.9 | 0.1336 |
| 4 | 82,585.6 | 0.1172 |
| 5 | 79,247.0 | 0.1005 |
| 6 | 76,404.9 | 0.1121 |
| 7 | 73,819.8 | 0.1102 |
| 8 | 71,917.6 | 0.1111 |

The best Silhouette within the evaluated range was obtained with K=3, at 0.1336. Although Inertia keeps decreasing as K increases, the selection prioritized the maximum observed Silhouette, avoiding adding segments without an equivalent improvement in internal separation.

### K-Means and Segment Profiles

| Cluster | Customers | Age | Balance | CreditScore | Products | Active | Card |
|---|---|---|---|---|---|---|---|
| 0 | 2,509 | 39.77 | 119,730 | 651.45 | 1.52 | 49.74% | 71.38% |
| 1 | 5,014 | 38.51 | 62,093 | 649.67 | 1.53 | 51.68% | 70.66% |
| 2 | 2,477 | 38.89 | 61,818 | 651.33 | 1.54 | 52.97% | 69.48% |

Cluster 0 holds 2,509 customers and is mainly distinguished by its average balance of about 119,730, compared with around 62,000 in the other two segments. Its churn rate reaches 32.44%, almost double the rates observed in Clusters 1 and 2. Cluster 1 is the largest group and has the lowest churn rate, 16.15%.

> **Figures (see PDF):** churn rate by K-Means cluster (bar chart) and K-Means projected onto the first two principal components.

The PCA visualization represents the three groups in two dimensions, but PC1 and PC2 together explain 25.37% of the variance. The visual separation should therefore be read as a projection of the original space and not as a complete representation.

## Dimensionality Reduction and DBSCAN

The full PCA showed that PC1 and PC2 explain 15.04% and 10.33% of the variance respectively, for a cumulative 25.37%. The first three components reach 35.32%. For DBSCAN, ten principal components were used, enough to reach a cumulative variance of 95.68%. t-SNE was applied on a sample of 3,000 records to provide an alternative visualization of the patterns.

### Variance Explained by PCA

| Component | Variance explained | Cumulative |
|---|---|---|
| PC1 | 15.04% | 15.04% |
| PC2 | 10.33% | 25.37% |
| PC3 | 9.95% | 35.32% |
| PC4 | 9.30% | 44.63% |
| PC5 | 9.16% | 53.78% |
| PC6 | 9.08% | 62.87% |
| PC7 | 9.00% | 71.86% |
| PC8 | 8.83% | 80.69% |
| PC9 | 8.09% | 88.78% |
| PC10 | 6.89% | 95.68% |
| PC11 | 4.32% | 100.00% |

DBSCAN was evaluated with `eps` between 0.7 and 1.5 and `min_samples` of 5, 10, 15 and 20. The selection criterion considered configurations with at least two clusters, prioritized those with less than 50% noise, and, within that set, chose the one with the highest Silhouette.

### Relevant DBSCAN Configurations

| eps | min_samples | Clusters | Noise | Silhouette |
|---|---|---|---|---|
| 1.2 | 20 | 28 | 67.00% | 0.1710 |
| 1.3 | 20 | 31 | 52.19% | 0.1889 |
| 1.4 | 20 | 34 | 38.53% | 0.1751 |
| 1.5 | 20 | 26 | 28.34% | 0.0908 |

The selected configuration was `eps=1.4` and `min_samples=20`. It found 34 clusters and classified 3,853 records as noise, equivalent to 38.53%. Its Silhouette was 0.1751. The `eps=1.3` configuration had a higher Silhouette, 0.1889, but exceeded the 50% noise limit defined by the project criterion, so it was not selected.

### Method Comparison

| Method | Clusters | Silhouette | Noise |
|---|---|---|---|
| K-Means | 3 | 0.1336 | 0% |
| DBSCAN | 34 | 0.1751 | 38.53% |

DBSCAN gets a higher Silhouette, but does so with a much more fragmented segmentation and a considerable share of noise. K-Means produces three simpler segments that are easier to communicate and covers all customers. Both results are therefore complementary: DBSCAN detects local structure and outliers, while K-Means offers a more manageable global segmentation.

> **Figures (see PDF):** Silhouette Score comparison between K-Means and DBSCAN, and DBSCAN visualized through a PCA projection.

## Relationship Between Segmentation and Churn

Once the labels were built, `Exited` was used exclusively to measure the churn rate within each segment. This makes it possible to study whether certain groups concentrate a higher proportion of attrition without confusing unsupervised segmentation with supervised prediction.

K-Means shows a clear difference: Cluster 0 reaches 32.44% churn, while Cluster 1 records 16.15% and Cluster 2 16.67%. The difference between the highest and lowest segment is 16.29 percentage points. The pattern matches the geographic distribution: Cluster 0 corresponds to customers in Germany, Cluster 1 to France and Cluster 2 to Spain.

DBSCAN produces a much more granular structure. The noise group (-1) contains 3,853 customers and has 29.54% churn. Among the identified clusters, Cluster 9 has the highest rate, 53.09%, followed by Cluster 7 with 45.83% and Cluster 6 with 35.14%. These rates should be read as descriptions of the groups obtained and not as evidence of causality.

| DBSCAN group | Customers | Churn | Rate |
|---|---|---|---|
| 9 | 243 | 129 | 53.09% |
| 7 | 216 | 99 | 45.83% |
| 6 | 148 | 52 | 35.14% |
| 16 | 327 | 99 | 30.28% |
| -1 (noise) | 3,853 | 1,138 | 29.54% |
| 8 | 478 | 111 | 23.22% |
| 24 | 138 | 30 | 21.74% |

The `Customer_Segment` variable was added to the final dataset as a feature derived from the clustering process. The segmented dataset keeps the `KMeans_Cluster` and `DBSCAN_Cluster` labels and adds `Customer_Segment` to make later use in analysis or business processes easier.

It is important to keep the methodological distinction: clusters were built without `Exited`; churn rates were computed afterward, as an analysis layer on the resulting labels.

## Reproducibility and File Structure

Execution was organized into three main scripts. Each works on the same input dataset and generates results that support the corresponding stage. Stage 4 documents these components so another person can rebuild the flow without relying on an oral explanation.

| File | Function |
|---|---|
| `Churn_Modelling.csv` | Input dataset with 10,000 records and 14 variables. |
| `1_EDA_RegresionLogistica.py` | EDA, preparation, train/test split, baseline Logistic Regression, metrics, confusion matrix, coefficients, Odds Ratios and confidence intervals. |
| `2_GradientBoosting_Optimizacion.py` | Model comparison, cross-validation, XGBoost Grid Search, feature importance, Stacking, confusion matrix and ROC/PR curves. |
| `3_AprendizajeNoSupervisado.py` | Clustering preparation, K-Means, PCA, profiles, DBSCAN, t-SNE, relationship with churn and segmented dataset. |

### Main Exported Results

The scripts generate CSV and PNG files as support.

- **Stage 1:** `resultados_metricas.csv`, `coeficientes_modelo.csv`, `odds_ratios_modelo.csv`, `intervalos_confianza.csv`.
- **Stage 2** (among others): `comparacion_final.csv`, `resultados_modelos.csv`, `resultados_cross_validation.csv`, `resultados_cross_validation_stacking.csv`, `mejores_parametros_xgboost.csv`, `resultado_stacking.csv`, feature importance files, `matriz_confusion_mejor_modelo.csv`, `comparacion_curvas_roc.png`, `comparacion_curvas_pr.png`.
- **Stage 3:** `evaluacion_kmeans.csv`, `cantidad_clientes_kmeans.csv`, `centroides_kmeans.csv`, `perfil_numerico_kmeans.csv`, `perfil_geografia_kmeans.csv`, `perfil_genero_kmeans.csv`, `churn_por_cluster.csv`, `evaluacion_dbscan.csv`, `churn_por_cluster_dbscan.csv`, `datos_preprocesados.csv`, `dataset_segmentado.csv`, `clientes_con_segmentacion.csv`, `resultado_tsne.csv` and the associated visualizations.

### Dependencies

The project uses Python and data analysis and Machine Learning libraries such as pandas, NumPy, Matplotlib, Seaborn, tabulate, scikit-learn, statsmodels, XGBoost, LightGBM and CatBoost. The scripts set `random_state=42` to keep the main random processes reproducible.

## Conclusions

The project built a progressive analytical solution to the bank churn problem. The exploratory analysis established the quality and structure of the dataset and showed that 20.37% of customers churn. Logistic Regression served as the baseline and showed a discrimination ability of 0.7748, but a low Recall for the positive class.

The second stage showed that boosting models substantially improve discrimination. Optimized XGBoost reached a ROC-AUC of 0.8689 and PR-AUC of 0.7201 on the test set, while Stacking obtained a ROC-AUC of 0.870 and PR-AUC of 0.724 in the comparative curves. The Stacking confusion matrix confirms an improvement in churn recovery over the baseline, although relevant false negatives remain.

The third stage brought a different perspective. K-Means built three interpretable segments and showed that Cluster 0 has 32.44% churn, versus 16.15% and 16.67% in the other segments. DBSCAN found a more granular structure, with 34 clusters and 38.53% of records considered noise. Its Silhouette was higher than K-Means, but the segmentation was more fragmented and harder to communicate.

Overall, the results should not be reduced to choosing a single algorithm. Supervised classification makes it possible to estimate the probability of belonging to the churn class, while segmentation describes groups of customers and shows how the attrition rate changes between them. Combining both perspectives provides a more complete base for later analyses.

The final result of Stage 4 is a traceable documentation of the project already built: it identifies the data, explains the methodology, keeps the main decisions, integrates the results, and organizes the execution path. The solution is ready to continue with business analysis, threshold tuning, evaluation of error costs, or exploitation of the segments, always clearly distinguishing between prediction, association and segmentation.

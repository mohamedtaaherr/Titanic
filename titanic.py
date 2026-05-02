# ============================================
# COMPLETE TITANIC GENDER SUBMISSION ANALYSIS
# Data Prep | EDA | Clustering | Classification | Regression
# ============================================

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, f1_score, roc_auc_score, \
    classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings;

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================
# 1. DATA PREPARATION
# ============================================
print("=" * 60);
print("1. DATA PREPARATION");
print("=" * 60)

# 1.1 Read & Understand - FIXED PATH
df = pd.read_csv(r"D:\Dataset\gender_submission.csv")  # Use raw string with r prefix
print(f"Shape: {df.shape} | Sample:\n{df.sample(5, random_state=42)}")

# 1.2 Basic Info
print(f"\nColumns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")
print(f"Missing values: {df.isnull().sum().sum()}")

# 1.3 Check Balance of Target
print(f"\nSurvival Distribution:")
print(df['Survived'].value_counts())
print(f"Survival Rate: {df['Survived'].mean() * 100:.2f}%")

# 1.4 Data Cleaning
df = df.drop_duplicates()
print(f"After removing duplicates: {len(df)} rows")

# 1.5 Fix Column Names
df.columns = df.columns.str.lower().str.strip()
print(f"Columns: {df.columns.tolist()}")

# 1.6 Outlier Flagging (IQR)
num_cols = ['passengerid', 'survived']
outliers = 0
for c in num_cols:
    Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    outliers += ((df[c] < Q1 - 1.5 * (Q3 - Q1)) | (df[c] > Q3 + 1.5 * (Q3 - Q1))).sum()
print(f"Outliers detected: {outliers}")

# ============================================
# 1.7 EDA - Univariate Analysis
# ============================================
print("\n" + "=" * 60);
print("1.7 EDA - UNIVARIATE");
print("=" * 60)

# Distribution plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# PassengerId distribution
axes[0].hist(df['passengerid'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_title('Passenger ID Distribution')
axes[0].set_xlabel('Passenger ID')
axes[0].set_ylabel('Frequency')

# Survival distribution (bar plot)
survival_counts = df['survived'].value_counts()
axes[1].bar(['Died (0)', 'Survived (1)'], survival_counts.values, color=['red', 'green'], edgecolor='black', alpha=0.7)
axes[1].set_title('Survival Distribution')
axes[1].set_ylabel('Count')
for i, v in enumerate(survival_counts.values):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('1_titanic_distributions.png')
plt.show()

# Skewness & Kurtosis
skew_kurt = pd.DataFrame({
    'Skewness': df[num_cols].skew(),
    'Kurtosis': df[num_cols].kurtosis()
})
print(f"\nSkewness & Kurtosis:\n{skew_kurt}")

# ============================================
# 1.8 EDA - Bivariate Analysis
# ============================================
print("\n" + "=" * 60);
print("1.8 EDA - BIVARIATE");
print("=" * 60)

# Since we only have PassengerId and Survived, let's create meaningful visualizations

# Survival by Passenger ID ranges
df['passenger_range'] = pd.cut(df['passengerid'], bins=10)
survival_by_range = df.groupby('passenger_range')['survived'].mean()

plt.figure(figsize=(12, 5))
survival_by_range.plot(kind='bar', color='coral', edgecolor='black', alpha=0.7)
plt.title('Survival Rate by Passenger ID Range')
plt.xlabel('Passenger ID Range')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('2_survival_by_range.png')
plt.show()

# Survival percentage pie chart
plt.figure(figsize=(8, 8))
colors = ['#ff6b6b', '#51cf66']
plt.pie(survival_counts.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0.05))
plt.title('Overall Survival Percentage')
plt.savefig('3_survival_pie.png')
plt.show()

# ============================================
# 1.9 EDA - Multivariate Analysis
# ============================================
print("\n" + "=" * 60);
print("1.9 EDA - MULTIVARIATE");
print("=" * 60)

# 3D Scatter Plot (PassengerId vs Index vs Survived)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['passengerid'], df.index, df['survived'],
           c=df['survived'], cmap='coolwarm', alpha=0.6, s=20)
ax.set_xlabel('Passenger ID')
ax.set_ylabel('Index')
ax.set_zlabel('Survived')
ax.set_title('3D Scatter: Passenger ID vs Index vs Survival')
plt.savefig('4_3d_scatter.png')
plt.show()

# VIF (Multicollinearity) - limited features
if len(num_cols) > 1:
    vif = pd.DataFrame({'Variable': num_cols,
                        'VIF': [variance_inflation_factor(df[num_cols].values, i) for i in range(len(num_cols))]})
    print(f"\nVariance Inflation Factor:\n{vif}")
else:
    print("\nVIF: Only one feature, skipping")

# PCA (Principal Component Analysis)
if len(num_cols) >= 2:
    pca = PCA(2).fit_transform(StandardScaler().fit_transform(df[num_cols]))
    df['pca1'], df['pca2'] = pca[:, 0], pca[:, 1]
    var_ratio = PCA(2).fit(StandardScaler().fit_transform(df[num_cols])).explained_variance_ratio_
    print(f"PCA explained variance: {var_ratio.sum():.2%}")

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['pca1'], df['pca2'], c=df['survived'], cmap='coolwarm', alpha=0.6, edgecolors='black')
    plt.colorbar(scatter, label='Survived')
    plt.title('PCA Visualization of Titanic Data')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig('5_pca.png')
    plt.show()

# ============================================
# 1.10 Data Preprocessing
# ============================================
print("\n" + "=" * 60);
print("1.10 PREPROCESSING");
print("=" * 60)

# Prepare features
features = ['passengerid']
X = df[features].copy()
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train size: {len(X_train)} ({len(X_train) / len(df) * 100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test) / len(df) * 100:.1f}%)")
print(f"Train survival rate: {y_train.mean() * 100:.2f}%")
print(f"Test survival rate: {y_test.mean() * 100:.2f}%")

# ============================================
# 2. CLUSTERING (Unsupervised)
# ============================================
print("\n" + "=" * 60);
print("2. CLUSTERING");
print("=" * 60)

X_clust = StandardScaler().fit_transform(X)

# Find optimal k (if enough data points)
if len(X_clust) >= 10:
    sil_scores = []
    k_range = range(2, min(11, len(X_clust) // 2))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_clust)
        if len(set(labels)) > 1:
            sil_scores.append(silhouette_score(X_clust, labels))
        else:
            sil_scores.append(-1)

    if sil_scores:
        opt_k = k_range[np.argmax(sil_scores)]
        labels = KMeans(n_clusters=opt_k, random_state=42, n_init=10).fit_predict(X_clust)
        sil = silhouette_score(X_clust, labels)
        dbi = davies_bouldin_score(X_clust, labels)
        print(f"Optimal k={opt_k} | Silhouette={sil:.4f} | DBI={dbi:.4f}")

        # Visualize clusters
        if 'pca1' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['pca1'], df['pca2'], c=labels, cmap='tab10', alpha=0.6)
            plt.colorbar(label='Cluster')
            plt.title(f'K-Means Clustering (k={opt_k})')
            plt.savefig('6_clustering.png')
            plt.show()
else:
    print("Insufficient data for clustering")

# ============================================
# 3. CLASSIFICATION
# ============================================
print("\n" + "=" * 60);
print("3. CLASSIFICATION");
print("=" * 60)

# Train multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results_cls = []
for name, clf in classifiers.items():
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, 'predict_proba') else y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5

    results_cls.append({'Model': name, 'Accuracy': acc, 'F1': f1, 'AUC': auc})
    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# Best classifier
best_clf = pd.DataFrame(results_cls).sort_values('Accuracy', ascending=False).iloc[0]
print(f"\nBEST CLASSIFIER: {best_clf['Model']} (Acc={best_clf['Accuracy']:.4f})")

# Classification Report for best model
best_model = classifiers[best_clf['Model']]
best_model.fit(X_train_s, y_train)
y_pred_best = best_model.predict(X_test_s)
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_best, target_names=['Died', 'Survived'])}")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.DataFrame({'feature': features, 'importance': best_model.feature_importances_})
    print(f"\nFeature Importance:\n{feat_imp}")
elif hasattr(best_model, 'coef_'):
    feat_imp = pd.DataFrame({'feature': features, 'coefficient': best_model.coef_[0]})
    print(f"\nFeature Coefficients:\n{feat_imp}")

# ============================================
# 4. REGRESSION (Option B)
# ============================================
print("\n" + "=" * 60);
print("4. REGRESSION (Option B)");
print("=" * 60)

# Regression models to predict survival probability
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

results_reg = []
for name, model in reg_models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    # Clip predictions to [0,1] range for probability
    y_pred_clipped = np.clip(y_pred, 0, 1)

    mae = mean_absolute_error(y_test, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
    r2 = r2_score(y_test, y_pred_clipped)

    results_reg.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2})
    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# Best regression model
best_reg = pd.DataFrame(results_reg).sort_values('R²', ascending=False).iloc[0]
print(f"\nBEST REGRESSION: {best_reg['Model']} (R²={best_reg['R²']:.4f})")

# Compare models visually
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
res_df = pd.DataFrame(results_reg)
for i, metric in enumerate(['MAE', 'RMSE', 'R²']):
    axes[i].bar(res_df['Model'], res_df[metric], color=['skyblue', 'lightcoral', 'lightgreen'][i], edgecolor='black')
    axes[i].set_title(f'{metric} Comparison')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('7_regression_comparison.png')
plt.show()

# Best model predictions vs actual
best_reg_model = reg_models[best_reg['Model']]
best_reg_model.fit(X_train_s, y_train)
y_pred_reg = np.clip(best_reg_model.predict(X_test_s), 0, 1)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_reg, alpha=0.5, edgecolors='black')
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Survival')
plt.ylabel('Predicted Probability')
plt.title(f'Best Regression: {best_reg["Model"]}\nR²={best_reg["R²"]:.4f}')
plt.legend()
plt.savefig('8_best_regression.png')
plt.show()

# Residual analysis
residuals = y_test - y_pred_reg
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(y_pred_reg, residuals, alpha=0.5, edgecolors='black')
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted');
axes[0].set_ylabel('Residuals');
axes[0].set_title('Residual Plot')
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1].set_title('Residual Distribution')
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot')
plt.tight_layout()
plt.savefig('9_residual_analysis.png')
plt.show()

# ============================================
# ADDITIONAL INSIGHTS
# ============================================
print("\n" + "=" * 60);
print("ADDITIONAL INSIGHTS");
print("=" * 60)

# Survival statistics
total_passengers = len(df)
survived = df['survived'].sum()
died = total_passengers - survived

print(f"""
TITANIC GENDER SUBMISSION ANALYSIS SUMMARY:
{'=' * 50}
Total Passengers: {total_passengers}
Survived: {survived} ({survived / total_passengers * 100:.2f}%)
Died: {died} ({died / total_passengers * 100:.2f}%)

CLASSIFICATION RESULTS:
- Best Model: {best_clf['Model']}
- Accuracy: {best_clf['Accuracy']:.4f}
- F1-Score: {best_clf['F1']:.4f}
- AUC: {best_clf['AUC']:.4f}

REGRESSION RESULTS:
- Best Model: {best_reg['Model']}
- R² Score: {best_reg['R²']:.4f}
- MAE: {best_reg['MAE']:.4f}
- RMSE: {best_reg['RMSE']:.4f}
""")

# Create a simple baseline prediction
baseline_pred = [df['survived'].mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f"Baseline MAE (always predict mean): {baseline_mae:.4f}")

# Improvement over baseline
if best_reg['MAE'] < baseline_mae:
    improvement = (baseline_mae - best_reg['MAE']) / baseline_mae * 100
    print(f"Best model improves MAE by {improvement:.1f}% over baseline")

print("\n" + "=" * 60)
print("✓ Analysis Complete! All 9 plots saved:")
print("  1_titanic_distributions.png")
print("  2_survival_by_range.png")
print("  3_survival_pie.png")
print("  4_3d_scatter.png")
print("  5_pca.png")
print("  6_clustering.png")
print("  7_regression_comparison.png")
print("  8_best_regression.png")
print("  9_residual_analysis.png")
print("=" * 60)
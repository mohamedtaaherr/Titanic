#  Titanic Survival Prediction - Complete ML Pipeline

##  Project Overview
End-to-end machine learning project analyzing the Titanic dataset. This project demonstrates a **complete data science workflow** from raw data to model evaluation.

---

##  WHAT THIS PROJECT DOES (Complete List)

### 1. DATA PREPARATION & CLEANING
| Task | Description |
|------|-------------|
| Load CSV data | Read Titanic dataset from file |
| Shape analysis | Display rows (891) and columns (12) |
| Random sampling | Show 5 random rows to understand data |
| Column identification | List all column names and data types |
| Missing value count | Identify 866 missing values across dataset |
| Target balance check | Calculate survival rate (38.38%) |
| Duplicate removal | Remove duplicate rows (0 found) |
| Column name standardization | Convert to lowercase with underscores |
| Outlier detection | IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR) |

### 2. EXPLORATORY DATA ANALYSIS (EDA)

#### Univariate Analysis
| Task | Output |
|------|--------|
| Passenger ID distribution | Histogram visualization |
| Survival distribution | Bar chart (Died: 549, Survived: 342) |
| Skewness calculation | Measure of distribution asymmetry |
| Kurtosis calculation | Measure of tail heaviness |

#### Bivariate Analysis
| Task | Output |
|------|--------|
| Survival by passenger range | Bar chart of survival rate across ID ranges |
| Survival pie chart | Percentage visualization (61.62% died, 38.38% survived) |

#### Multivariate Analysis
| Task | Output |
|------|--------|
| 3D scatter plot | Passenger ID vs Index vs Survival |
| VIF calculation | Variance Inflation Factor (multicollinearity check) |
| PCA transformation | Reduce to 2 components (100% variance explained) |
| PCA visualization | 2D scatter plot colored by survival |

### 3. DATA PREPROCESSING
| Task | Description |
|------|-------------|
| Feature selection | Use 'passengerid' as feature |
| Target definition | 'survived' as target variable |
| Train/test split | 80/20 split with stratification |
| Stratification | Maintain same survival rate in both sets |
| Feature scaling | StandardScaler (mean=0, variance=1) |

### 4. CLUSTERING (Unsupervised Learning)
| Task | Description |
|------|-------------|
| Optimal k finding | Test k from 2 to 10 |
| Silhouette score | Measure cluster separation quality |
| Davies-Bouldin index | Measure cluster similarity |
| K-Means clustering | Final model with optimal k=2 |
| Cluster visualization | Scatter plot colored by cluster |

### 5. CLASSIFICATION (Supervised Learning)
| Task | Description |
|------|-------------|
| Logistic Regression | Train classifier with probability output |
| Random Forest | Train ensemble classifier (100 trees) |
| Accuracy score | Percentage of correct predictions |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve |
| Classification report | Precision, recall, f1-score per class |
| Feature coefficients | Extract model coefficients/importance |
| Best model selection | Compare models and pick best (Logistic Regression: 61.45%) |

### 6. REGRESSION ANALYSIS (Option B)
| Task | Description |
|------|-------------|
| Linear Regression | Basic regression model |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization |
| Random Forest Regressor | Ensemble regression (100 trees) |
| MAE calculation | Mean Absolute Error |
| RMSE calculation | Root Mean Square Error |
| R² calculation | Coefficient of determination |
| Model comparison | Bar charts comparing all 4 models |
| Best model selection | Lasso (R² = -0.000) |

### 7. MODEL EVALUATION & DIAGNOSTICS
| Task | Description |
|------|-------------|
| Residual calculation | Actual - Predicted values |
| Residual plot | Scatter plot of predicted vs residuals |
| Residual histogram | Distribution of residuals |
| Q-Q plot | Normality check (quantile-quantile) |
| Baseline comparison | Compare model to always-predict-mean baseline |

### 8. VISUALIZATIONS GENERATED (9 Plots)
| File | What It Shows |
|------|----------------|
| `1_titanic_distributions.png` | Passenger ID histogram + Survival bar chart |
| `2_survival_by_range.png` | Survival rate across passenger ID ranges |
| `3_survival_pie.png` | Overall survival percentage (62% vs 38%) |
| `4_3d_scatter.png` | 3D view: Pass ID vs Index vs Survival |
| `5_pca.png` | PCA 2D projection colored by survival |
| `6_clustering.png` | K-Means clusters (k=2) in PCA space |
| `7_regression_comparison.png` | MAE, RMSE, R² for all 4 models |
| `8_best_regression.png` | Predicted vs actual survival probabilities |
| `9_residual_analysis.png` | Residual plot, histogram, Q-Q plot |

---

##  Dataset Information
| Property | Value |
|----------|-------|
| **Source** | Kaggle Titanic Dataset |
| **Samples** | 891 passengers |
| **Features** | 12 (mix of numerical, categorical, text) |
| **Target** | Survived (0 = Died, 1 = Survived) |
| **Survival Rate** | 38.38% |

---

##  DATA CLEANING DOCUMENTATION (For Job Interviews)

### Step-by-Step Cleaning Process

```python
# Step 1: Load and inspect
df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape)  # (891, 12)

# Step 2: Check for duplicates
df = df.drop_duplicates()  # 0 duplicates found

# Step 3: Standardize column names
df.columns = df.columns.str.lower().str.strip()
# 'PassengerId' → 'passengerid'
# 'Survived' → 'survived'

# Step 4: Outlier detection (IQR method)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR))
    print(f"{col}: {outliers.sum()} outliers")

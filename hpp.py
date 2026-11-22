
# ============================================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ============================================================

# STEP 0: IMPORT LIBRARIES

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STEP 1: DATA INSPECTION
# ============================================================

# Load house price dataset
df = pd.read_csv("train.csv")

print("\n========== DATA INSPECTION ==========")

# First five rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Dataset information
print("\nDataset information:")
df.info()

# Statistical summary
print("\nStatistical description:")
print(df.describe())

# Missing values
print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], kde=True)
plt.title("House Price Distribution")
plt.xlabel("Sale Price")
plt.show()


# ============================================================
# STEP 2: REMOVE UNWANTED VARIABLES
# ============================================================

print("\n========== REMOVE UNWANTED VARIABLES ==========")

# Remove duplicate rows
df = df.drop_duplicates()

# Separate target variable
y = df["SalePrice"]

# Remove target from input features
X = df.drop(columns=["SalePrice"])

# Remove ID column
X = X.drop(columns=["Id"], errors="ignore")

print("\nFeatures after removing ID:")
print(X.shape)


# ============================================================
# STEP 3: HANDLE NULL / INAPPROPRIATE VALUES
# ============================================================

print("\n========== HANDLE MISSING VALUES ==========")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Some categorical columns contain missing values
# because a house may not have that facility.
# We will handle them using imputation later.

# Remove columns having more than 80% missing values.
missing_percentage = X.isnull().mean()

unwanted_columns = missing_percentage[
    missing_percentage > 0.80
].index.tolist()

X = X.drop(columns=unwanted_columns)

print("\nRemoved columns:")
print(unwanted_columns)

print("\nRemaining features:", X.shape[1])


# ============================================================
# STEP 4: FEATURE GENERATION & TRANSFORMATION
# ============================================================

print("\n========== FEATURE ENGINEERING ==========")

# Total house area
X["TotalSF"] = (
    X["TotalBsmtSF"].fillna(0) +
    X["1stFlrSF"].fillna(0) +
    X["2ndFlrSF"].fillna(0)
)

# Total bathrooms
X["TotalBath"] = (
    X["FullBath"].fillna(0) +
    0.5 * X["HalfBath"].fillna(0) +
    X["BsmtFullBath"].fillna(0) +
    0.5 * X["BsmtHalfBath"].fillna(0)
)

# Age of house at the time of sale
X["HouseAge"] = (
    X["YrSold"] - X["YearBuilt"]
).clip(lower=0)

# Years since last remodeling
X["RemodAge"] = (
    X["YrSold"] - X["YearRemodAdd"]
).clip(lower=0)

# Total porch area
porch_columns = [
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "WoodDeckSF"
]

X["TotalPorchSF"] = X[porch_columns].fillna(0).sum(axis=1)

# Binary features
X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)

X["HasBasement"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)

X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)

# Log transformation of skewed numerical features
for col in ["LotArea", "TotalSF", "GrLivArea"]:
    if col in X.columns:
        X[col] = np.log1p(X[col].clip(lower=0))

print("\nNew features created:")
print([
    "TotalSF",
    "TotalBath",
    "HouseAge",
    "RemodAge",
    "TotalPorchSF",
    "HasGarage",
    "HasBasement",
    "HasPool"
])

print("\nUpdated dataset shape:", X.shape)


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

# Split before learning imputation, outlier limits,
# and scaling parameters to avoid data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# STEP 5: OUTLIER TREATMENT
# ============================================================

print("\n========== OUTLIER TREATMENT ==========")

# Select numerical columns
numeric_columns = X_train.select_dtypes(
    include=np.number
).columns

# Calculate lower and upper limits using training data only
lower_limits = X_train[numeric_columns].quantile(0.01)
upper_limits = X_train[numeric_columns].quantile(0.99)

# Apply the same limits to train and test datasets
X_train[numeric_columns] = X_train[numeric_columns].clip(
    lower=lower_limits,
    upper=upper_limits,
    axis=1
)

X_test[numeric_columns] = X_test[numeric_columns].clip(
    lower=lower_limits,
    upper=upper_limits,
    axis=1
)

print("Outlier treatment completed using 1st and 99th percentiles.")


# ============================================================
# STEP 6: SCALING OF FEATURES
# ============================================================

print("\n========== FEATURE SCALING ==========")

# Identify numerical and categorical features
numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Missing"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# STEP 7: BUILD & TRAIN MACHINE LEARNING MODELS
# ============================================================

print("\n========== MODEL TRAINING ==========")

# Define five machine learning regression algorithms
models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=100,
        epsilon=0.1,
        gamma="scale"
    )
}

# Store evaluation results
results = []

# Store trained models
trained_models = {}

for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    # Create complete machine learning pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    # Apply log transformation to target during training
    # and convert predictions back to original house prices
    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict house prices
    predictions = model.predict(X_test)

    # Store trained model
    trained_models[name] = model

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"{name} training completed.")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")


# ============================================================
# STEP 8: EVALUATE & COMPARE MODELS
# ============================================================

print("\n========== MODEL COMPARISON ==========")

# Convert results into DataFrame
comparison = pd.DataFrame(results)

# Sort by RMSE (lowest is best)
comparison = comparison.sort_values(
    by="RMSE",
    ascending=True
)

print("\nFinal Model Comparison:")
print(comparison.to_string(index=False))

# Identify best model
best_model_name = comparison.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# ============================================================
# VISUALIZATION 1: MODEL RMSE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="RMSE"
)

plt.title("RMSE Comparison of Regression Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 2: MODEL R2 SCORE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="R2 Score"
)

plt.title("R2 Score Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZATION 3: ACTUAL VS PREDICTED HOUSE PRICES
# ============================================================

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title(f"Actual vs Predicted: {best_model_name}")

min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--"
)

plt.tight_layout()
plt.show()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n========== FINAL RESULT ==========")

print("Best Model:", best_model_name)

print("Best RMSE:",
      comparison.iloc[0]["RMSE"])

print("Best MAE:",
      comparison.iloc[0]["MAE"])

print("Best R2 Score:",
      comparison.iloc[0]["R2 Score"])

# Display sample predictions
prediction_table = pd.DataFrame({
    "Actual Price": y_test.iloc[:10].values,
    "Predicted Price": best_predictions[:10]
})

print("\nSample Predictions:")
print(prediction_table)

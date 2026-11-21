"""
House Price Prediction with TensorFlow/Keras
============================================
Dataset: Kaggle House Prices - Advanced Regression Techniques
Task: Regression (predict SalePrice)

Pipeline:
 1. Data inspection
 2. Remove unwanted variables
 3. Handle null / inappropriate values
 4. Feature generation & transformation
 5. Outlier treatment
 6. Scaling of features
 7. Build & train TensorFlow regression models
 8. Evaluate & compare
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. DATA INSPECTION
# ---------------------------------------------------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test  shape:", test.shape)
print("\nTarget summary:\n", train["SalePrice"].describe())

# ---------------------------------------------------------------------
# 2. REMOVE UNWANTED VARIABLES
# ---------------------------------------------------------------------
# 'Id' is not predictive; keep 'Id' separately for submission
train_ids = train["Id"].copy()
test_ids  = test["Id"].copy()

train = train.drop(columns=["Id"])
test  = test.drop(columns=["Id"])

y = train["SalePrice"].copy()
X = train.drop(columns=["SalePrice"])

# ---------------------------------------------------------------------
# 3. FEATURE ENGINEERING & TRANSFORMATION (1.4)
# ---------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Total living area
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    # Total bathrooms
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    # Total porch area
    df["TotalPorch"] = (
        df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )

    # Property age
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["GarageAge"] = df["GarageAge"].where(df["GarageAge"] >= 0, np.nan)

    # Total rooms
    df["TotalRooms"] = (
        df["TotRmsAbvGrd"].fillna(0) + df["BedroomAbvGr"].fillna(0)
    )

    return df

X    = engineer_features(X)
test = engineer_features(test)

# ---------------------------------------------------------------------
# 4. NULL / INAPPROPRIATE VALUES + OUTLIER TREATMENT (1.3, 1.5)
# ---------------------------------------------------------------------
# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove extreme target outliers (Grubbs-style rule / IQR)
q1, q3 = y.quantile([0.25, 0.75])
iqr = q3 - q1
mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
print(f"Removing {len(y) - mask.sum()} SalePrice outliers")
X, y = X.loc[mask], y.loc[mask]

# Log-transform target (helps regression stability)
y_log = np.log1p(y)

# ---------------------------------------------------------------------
# 5. PREPROCESSING PIPELINE (impute + encode + scale) - 3.1
# ---------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])

# Fit on train, transform both train & test
X_prep    = preprocessor.fit_transform(X)
test_prep = preprocessor.transform(test)

print("Processed feature matrix:", X_prep.shape)

# ---------------------------------------------------------------------
# 6. TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_prep, y_log.values, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 7. TENSORFLOW MODELS
# ---------------------------------------------------------------------
def build_dnn_model(input_dim):
    """Deep Neural Network for regression."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),  # linear output for regression
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

def build_mlp_model(input_dim):
    """Light MLP (LightGBM-style shallow alternative)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

# ---------------------------------------------------------------------
# 8. TRAIN DNN
# ---------------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    ),
]

dnn = build_dnn_model(X_tr.shape[1])
history = dnn.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

# ---------------------------------------------------------------------
# 9. EVALUATION (5.1)
# ---------------------------------------------------------------------
def evaluate(model, X_, y_log_, name="Model"):
    y_pred_log = model.predict(X_, verbose=0).ravel()
    y_pred = np.expm1(y_pred_log)          # back to original price
    y_true = np.expm1(y_log_)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n--- {name} ---")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    return {"model": name, "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

results = [evaluate(dnn, X_val, y_val, "TensorFlow DNN")]

# Optional: train MLP for comparison
mlp = build_mlp_model(X_tr.shape[1])
mlp.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=200, batch_size=64,
    callbacks=callbacks, verbose=0,
)
results.append(evaluate(mlp, X_val, y_val, "TensorFlow MLP"))

# K-Fold cross-validation option (for model stability)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []
for train_idx, val_idx in kf.split(X_prep):
    m = build_dnn_model(X_prep.shape[1])
    m.fit(
        X_prep[train_idx], y_log.values[train_idx],
        epochs=100, batch_size=64, verbose=0,
    )
    preds = np.expm1(m.predict(X_prep[val_idx], verbose=0).ravel())
    cv_r2.append(r2_score(np.expm1(y_log.values[val_idx]), preds))
print(f"\n5-Fold CV R² (DNN): {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

# ---------------------------------------------------------------------
# 10. TRAINING CURVES
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch"); plt.ylabel("MSE (log target)")
plt.title("Training vs Validation Loss")
plt.legend(); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
# 11. FINAL PREDICTION ON UNSEEN TEST SET + SAVE
# ---------------------------------------------------------------------
test_pred_log = dnn.predict(test_prep, verbose=0).ravel()
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred,
})
submission.to_csv("submission_tensorflow.csv", index=False)
print("\n✅ Saved predictions to submission_tensorflow.csv")

# Save model
dnn.save("house_price_dnn.keras")
print("✅ Model saved to house_price_dnn.keras")

"""
03_gene_level_ML.py

Gene-level machine-learning analysis.

Input:
    data/MASTER_AE16_AE_50.csv

Predictors:
    34 expression features + 16 AE latent features

Target:
    is_protein_coding
    1 = protein-coding
    0 = non-protein-coding

Models:
    XGBoost
    Random Forest
    KNN
    Decision Tree

Validation:
    10-fold KFold, shuffle=True, random_state=42
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
N_FOLDS = 10

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "MASTER_AE16_AE_50.csv"
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Required file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("=" * 70)
    print("GENE-LEVEL MACHINE LEARNING")
    print("=" * 70)

    print("Input:", INPUT_FILE)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # Original table layout:
    # predictors = all columns except final binary target
    # target     = final column

    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df.iloc[:, -1], errors="raise").astype(int).values

    feature_names = [str(c) for c in X_df.columns]
    X = X_df.values.astype(np.float32)

    unique = np.unique(y)

    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(f"Expected binary target 0/1. Found: {unique}")

    print("\nFEATURE MATRIX")
    print("Predictors:", X.shape[1])
    print("Expected: 50 = 34 expression + 16 AE latent")

    print("\nTARGET")
    print("Target column:", df.columns[-1])
    print("Class 0:", int((y == 0).sum()))
    print("Class 1:", int((y == 1).sum()))

    return df, X, y, feature_names


# ============================================================
# 3. ORIGINAL MODELS
# ============================================================

def get_models():

    return {
        "XGB": XGBClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=300,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist"
        ),

        "RF": RandomForestClassifier(
            n_estimators=100,
            random_state=0,
            n_jobs=-1
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=3
        ),

        "DT": DecisionTreeClassifier(
            random_state=0
        )
    }


# ============================================================
# 4. 10-FOLD OUT-OF-FOLD VALIDATION
# ============================================================

def run_cross_validation(X, y):

    cv = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    models = get_models()

    metric_rows = []

    oof = pd.DataFrame({
        "observed": y
    })

    print("\n" + "=" * 70)
    print("10-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for name, model in models.items():

        print(f"\nRunning {name} ...")

        probability = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None
        )[:, 1]

        prediction = (probability >= 0.5).astype(int)

        auc = roc_auc_score(y, probability)
        accuracy = accuracy_score(y, prediction)
        precision = precision_score(y, prediction, zero_division=0)
        recall = recall_score(y, prediction, zero_division=0)
        f1 = f1_score(y, prediction, zero_division=0)

        metric_rows.append({
            "Model": name,
            "AUC": auc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        oof[f"{name}_probability"] = probability
        oof[f"{name}_prediction"] = prediction

        print(
            f"{name} | "
            f"AUC={auc:.4f} | "
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    metrics = pd.DataFrame(metric_rows)

    return metrics, oof


# ============================================================
# 5. TRAIN FINAL XGBOOST
# ============================================================

def train_final_xgb(X, y):

    print("\n" + "=" * 70)
    print("TRAIN FINAL XGBOOST ON FULL DATASET")
    print("=" * 70)

    model = XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        n_estimators=300,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X, y)

    print("[OK] Final XGBoost fitted.")

    return model


# ============================================================
# 6. SAVE RESULTS
# ============================================================

def save_results(metrics, oof):

    metrics_file = RESULT_DIR / "gene_level_ML_metrics.csv"
    oof_file = RESULT_DIR / "gene_level_ML_OOF.csv"

    metrics.to_csv(metrics_file, index=False)
    oof.to_csv(oof_file, index=False)

    print("\nSaved:", metrics_file)
    print("Saved:", oof_file)

    return metrics_file, oof_file


# ============================================================
# 7. SAVE FINAL XGBOOST MODEL
# ============================================================

def save_xgb_model(model, feature_names, y):

    model_file = MODEL_DIR / "xgb_model.json"
    metadata_file = MODEL_DIR / "xgb_metadata.json"

    model.save_model(model_file)

    metadata = {
        "model": "XGBoost",
        "analysis": "gene-level protein-coding classification",
        "target": "is_protein_coding",
        "class_0": "non-protein-coding",
        "class_1": "protein-coding",
        "n_features": len(feature_names),
        "features": feature_names,
        "n_samples": int(len(y)),
        "class_0_n": int((y == 0).sum()),
        "class_1_n": int((y == 1).sum()),
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 300,
        "tree_method": "hist",
        "cv": "10-fold KFold",
        "cv_shuffle": True,
        "cv_random_state": SEED
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved:", model_file)
    print("Saved:", metadata_file)

    return model_file, metadata_file


# ============================================================
# 8. VERIFY MODEL
# ============================================================

def verify_model(model_file, X):

    test_model = XGBClassifier()
    test_model.load_model(model_file)

    test_probability = test_model.predict_proba(X[:5])[:, 1]

    print("\nMODEL RELOAD TEST")
    print("[OK] XGBoost model reloaded.")
    print("First five probabilities:")
    print(test_probability)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    df, X, y, feature_names = load_data()

    metrics, oof = run_cross_validation(X, y)

    metrics_file, oof_file = save_results(
        metrics,
        oof
    )

    final_xgb = train_final_xgb(
        X,
        y
    )

    model_file, metadata_file = save_xgb_model(
        final_xgb,
        feature_names,
        y
    )

    verify_model(
        model_file,
        X
    )

    print("\n" + "=" * 70)
    print("GENE-LEVEL ML COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nFINAL METRICS")
    print(metrics.to_string(index=False))

    print("\nOUTPUT FILES")
    print(metrics_file)
    print(oof_file)
    print(model_file)
    print(metadata_file)

    print("\nDONE.")


if __name__ == "__main__":
    main()
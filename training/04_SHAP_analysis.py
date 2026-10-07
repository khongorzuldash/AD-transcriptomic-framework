"""
04_SHAP_analysis.py

SHAP analysis for the final gene-level XGBoost model.

Input:
    data/MASTER_AE16_AE_50.csv
    models/xgb_model.json

Outputs:
    results/SHAP_feature_importance.csv
    results/SHAP_values.csv
    results/SHAP_global_bar.png
    results/SHAP_beeswarm.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "MASTER_AE16_AE_50.csv"
MODEL_FILE = ROOT / "models" / "xgb_model.json"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df.iloc[:, -1], errors="raise").astype(int).values

    feature_names = [str(c) for c in X_df.columns]
    X = X_df.values.astype(np.float32)

    print("=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)
    print("Genes:", len(df))
    print("Features:", X.shape[1])
    print("Target:", df.columns[-1])

    return df, X_df, X, y, feature_names


# ============================================================
# 3. LOAD FINAL XGBOOST
# ============================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_FILE}\n"
            "Run training/03_gene_level_ML.py first."
        )

    model = XGBClassifier()
    model.load_model(MODEL_FILE)

    print("[OK] XGBoost model loaded.")

    return model


# ============================================================
# 4. CALCULATE SHAP VALUES
# ============================================================

def calculate_shap(model, X_df):

    print("\nCalculating SHAP values ...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    print("[OK] SHAP matrix:", shap_values.shape)

    return shap_values


# ============================================================
# 5. GLOBAL SHAP IMPORTANCE
# ============================================================

def calculate_importance(shap_values, feature_names):

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
        "sum_SHAP": shap_values.sum(axis=0),
        "sum_abs_SHAP": np.abs(shap_values).sum(axis=0)
    })

    importance = importance.sort_values(
        "mean_abs_SHAP",
        ascending=False
    ).reset_index(drop=True)

    return importance


# ============================================================
# 6. SAVE TABLES
# ============================================================

def save_tables(shap_values, importance, feature_names):

    importance_file = RESULT_DIR / "SHAP_feature_importance.csv"
    values_file = RESULT_DIR / "SHAP_values.csv"

    importance.to_csv(importance_file, index=False)

    pd.DataFrame(
        shap_values,
        columns=feature_names
    ).to_csv(values_file, index=False)

    print("\nSaved:", importance_file)
    print("Saved:", values_file)

    return importance_file, values_file


# ============================================================
# 7. SHAP GLOBAL BAR PLOT
# ============================================================

def save_global_bar(model, X_df):

    plt.figure()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap.summary_plot(
        shap_values,
        X_df,
        plot_type="bar",
        max_display=20,
        show=False
    )

    plt.tight_layout()

    output = RESULT_DIR / "SHAP_global_bar.png"

    plt.savefig(
        output,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output)

    return output


# ============================================================
# 8. SHAP BEESWARM
# ============================================================

def save_beeswarm(model, X_df):

    plt.figure()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap.summary_plot(
        shap_values,
        X_df,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    output = RESULT_DIR / "SHAP_beeswarm.png"

    plt.savefig(
        output,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output)

    return output


# ============================================================
# 9. MAIN
# ============================================================

def main():

    df, X_df, X, y, feature_names = load_data()

    model = load_model()

    shap_values = calculate_shap(
        model,
        X_df
    )

    importance = calculate_importance(
        shap_values,
        feature_names
    )

    importance_file, values_file = save_tables(
        shap_values,
        importance,
        feature_names
    )

    global_bar = save_global_bar(
        model,
        X_df
    )

    beeswarm = save_beeswarm(
        model,
        X_df
    )

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nTOP 20 FEATURES")
    print(
        importance.head(20).to_string(
            index=False
        )
    )

    print("\nOUTPUT FILES")
    print(importance_file)
    print(values_file)
    print(global_bar)
    print(beeswarm)

    print("\nDONE.")


if __name__ == "__main__":
    main()
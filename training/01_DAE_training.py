"""
01_DAE_training.py

Training script for the denoising autoencoder (DAE) used for
gene-level reconstruction-error filtering of the human
DESeq2-processed RNA-seq expression matrix.

Input:
    D:/NATURE/New/24k_34_deseq2.csv

Outputs:
    results/DAE_training_validation_loss.csv
    results/DAE_training_validation_loss.png
    results/DAE_all_genes_with_flags.csv
    results/DAE_clean_genes.csv
    results/DAE_outlier_genes.csv

    models/dae_model.pt
    models/dae_scaler.pkl
    models/dae_metadata.json
"""

from pathlib import Path
import re
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
LATENT_DIM = 2
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 80
NOISE_STD = 0.2
OUTLIER_TOP_PERCENT = 5

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = Path(r"D:/NATURE/New/24k_34_deseq2.csv")
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. DAE MODEL
# ============================================================

class DAE(nn.Module):
    """Denoising autoencoder: 34 -> 64 -> 2 -> 64 -> 34."""

    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, input_dim))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ============================================================
# 3. SAFE CSV READER
# ============================================================

def safe_read_csv(path):
    """Read CSV using common encodings and delimiter detection."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python", low_memory=False)
            if df.shape[1] > 1:
                print(f"[OK] CSV loaded: encoding={enc}, delimiter=auto")
                return df
        except Exception as e:
            last_error = e

    for enc in encodings:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    print(f"[OK] CSV loaded: encoding={enc}, separator={repr(sep)}")
                    return df
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Could not read input CSV. Last error: {last_error}")


# ============================================================
# 4. LOAD AND VALIDATE DATA
# ============================================================

def load_expression_data(path):
    """Load the DESeq2-processed matrix and identify 34 SRR columns."""

    df = safe_read_csv(path)

    print("\nDataset")
    print("Genes:", df.shape[0])
    print("Columns:", df.shape[1])

    sample_pattern = re.compile(r"^SRR\d+")
    expr_cols = [c for c in df.columns if sample_pattern.match(str(c))]

    print("Detected SRR expression columns:", len(expr_cols))

    if len(expr_cols) != 34:
        raise ValueError(f"Expected 34 SRR expression columns, found {len(expr_cols)}.")

    df[expr_cols] = df[expr_cols].apply(pd.to_numeric, errors="coerce")

    missing = int(df[expr_cols].isna().sum().sum())

    if missing > 0:
        raise ValueError(f"Expression matrix contains {missing} missing/non-numeric values.")

    X_raw = df[expr_cols].values.astype(np.float32)

    if np.any(X_raw < 0):
        raise ValueError("Negative expression values detected before log1p.")

    return df, expr_cols, X_raw


# ============================================================
# 5. PREPROCESSING
# ============================================================

def preprocess_expression(X_raw):
    """Apply log1p transformation followed by StandardScaler."""

    X_log = np.log1p(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log).astype(np.float32)

    if not np.isfinite(X_scaled).all():
        raise ValueError("NaN or infinite values detected after preprocessing.")

    return X_log, X_scaled, scaler


# ============================================================
# 6. TRAIN DAE
# ============================================================

def train_dae(X_scaled):
    """Train DAE using an 80:20 train-validation split."""

    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=SEED)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_scaled.shape[1]

    model = DAE(input_dim=input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    print("\nTraining DAE")
    print("Device:", DEVICE)
    print("Training genes:", len(X_train))
    print("Validation genes:", len(X_val))
    print("Input dimension:", input_dim)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0

        for (batch,) in train_loader:
            clean_x = batch.to(DEVICE)
            noisy_x = clean_x + NOISE_STD * torch.randn_like(clean_x)

            optimizer.zero_grad()
            recon, _ = model(noisy_x)
            loss = criterion(recon, clean_x)
            loss.backward()
            optimizer.step()

            train_total += loss.item() * clean_x.size(0)

        train_loss = train_total / len(train_loader.dataset)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for (batch,) in val_loader:
                clean_x = batch.to(DEVICE)
                recon, _ = model(clean_x)
                loss = criterion(recon, clean_x)
                val_total += loss.item() * clean_x.size(0)

        val_loss = val_total / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f}")

    return model, train_losses, val_losses


# ============================================================
# 7. RECONSTRUCTION ERROR
# ============================================================

def calculate_reconstruction(model, X_scaled):
    """Calculate gene-wise reconstruction errors and latent coordinates."""

    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        reconstructed, latent = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        latent = latent.cpu().numpy()

    return errors, latent


# ============================================================
# 8. SAVE TRAINING CURVE
# ============================================================

def save_training_curve(train_losses, val_losses):
    """Save training/validation losses as CSV and PNG."""

    loss_df = pd.DataFrame({
        "epoch": np.arange(1, EPOCHS + 1),
        "train_loss": train_losses,
        "validation_loss": val_losses
    })

    loss_csv = RESULT_DIR / "DAE_training_validation_loss.csv"
    loss_png = RESULT_DIR / "DAE_training_validation_loss.png"

    loss_df.to_csv(loss_csv, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Training")
    plt.plot(loss_df["epoch"], loss_df["validation_loss"], label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title("DAE training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png, dpi=600, bbox_inches="tight")
    plt.close()

    return loss_csv, loss_png


# ============================================================
# 9. SAVE MODEL AND METADATA
# ============================================================

def save_model_files(model, scaler, expr_cols, threshold, n_genes, n_clean, n_outliers):
    """Save pretrained model, fitted scaler, and model metadata."""

    model_file = MODEL_DIR / "dae_model.pt"
    scaler_file = MODEL_DIR / "dae_scaler.pkl"
    metadata_file = MODEL_DIR / "dae_metadata.json"

    torch.save(model.state_dict(), model_file)
    joblib.dump(scaler, scaler_file)

    metadata = {
        "model_name": "Human_DAE",
        "input_dataset": "24k_34_deseq2.csv",
        "input_description": "DESeq2-processed human RNA-seq expression matrix",
        "architecture": "34-64-2-64-34",
        "input_dim": len(expr_cols),
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "activation": "ReLU",
        "noise_type": "Gaussian",
        "noise_std": NOISE_STD,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "MSE",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_validation_split": "80:20",
        "seed": SEED,
        "preprocessing": ["log1p", "StandardScaler"],
        "outlier_method": "Top reconstruction-error percentile",
        "outlier_top_percent": OUTLIER_TOP_PERCENT,
        "outlier_threshold": float(threshold),
        "n_input_genes": int(n_genes),
        "n_retained_genes": int(n_clean),
        "n_outlier_genes": int(n_outliers),
        "expression_columns": [str(c) for c in expr_cols]
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return model_file, scaler_file, metadata_file


# ============================================================
# 10. VERIFY SAVED MODEL
# ============================================================

def verify_saved_files(model_file, scaler_file, input_dim):
    """Reload saved model and scaler to verify that both are valid."""

    test_model = DAE(input_dim=input_dim)
    test_model.load_state_dict(torch.load(model_file, map_location="cpu"))
    test_model.eval()

    test_scaler = joblib.load(scaler_file)

    if test_scaler.n_features_in_ != input_dim:
        raise ValueError("Saved scaler feature dimension does not match DAE input dimension.")

    print("[OK] Saved DAE model successfully reloaded.")
    print("[OK] Saved StandardScaler successfully reloaded.")


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("HUMAN DAE TRAINING PIPELINE")
    print("=" * 70)
    print("Input:", INPUT_FILE)
    print("Repository root:", ROOT)

    # Load DESeq2-processed expression matrix
    df, expr_cols, X_raw = load_expression_data(INPUT_FILE)

    # log1p + StandardScaler
    _, X_scaled, scaler = preprocess_expression(X_raw)

    # Train DAE
    model, train_losses, val_losses = train_dae(X_scaled)

    # Save training curve
    loss_csv, loss_png = save_training_curve(train_losses, val_losses)

    # Reconstruct all genes
    errors, latent = calculate_reconstruction(model, X_scaled)

    # Original top-5% reconstruction-error threshold
    threshold = np.percentile(errors, 100 - OUTLIER_TOP_PERCENT)
    outlier_flags = errors >= threshold

    # Add DAE results
    result = df.copy()
    result["DAE_reconstruction_error"] = errors
    result["DAE_outlier"] = outlier_flags
    result["DAE_z1"] = latent[:, 0]
    result["DAE_z2"] = latent[:, 1]

    clean = result.loc[~result["DAE_outlier"]].copy()
    outliers = result.loc[result["DAE_outlier"]].copy()

    # Save analysis tables
    all_file = RESULT_DIR / "DAE_all_genes_with_flags.csv"
    clean_file = RESULT_DIR / "DAE_clean_genes.csv"
    outlier_file = RESULT_DIR / "DAE_outlier_genes.csv"

    result.to_csv(all_file, index=False)
    clean.to_csv(clean_file, index=False)
    outliers.to_csv(outlier_file, index=False)

    # Save pretrained model/scaler/metadata
    model_file, scaler_file, metadata_file = save_model_files(
        model=model,
        scaler=scaler,
        expr_cols=expr_cols,
        threshold=threshold,
        n_genes=len(result),
        n_clean=len(clean),
        n_outliers=len(outliers)
    )

    # Verify saved files
    verify_saved_files(model_file, scaler_file, len(expr_cols))

    # Final report
    print("\n" + "=" * 70)
    print("DAE TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nDATA")
    print("Input genes:", len(result))
    print("Expression columns:", len(expr_cols))

    print("\nMODEL")
    print("Architecture: 34 -> 64 -> 2 -> 64 -> 34")
    print("Noise SD:", NOISE_STD)
    print("Learning rate:", LEARNING_RATE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    print("\nFILTERING")
    print("Top reconstruction-error percentage:", OUTLIER_TOP_PERCENT)
    print("Threshold:", threshold)
    print("Retained genes:", len(clean))
    print("Outlier genes:", len(outliers))

    print("\nMODEL FILES")
    print(model_file)
    print(scaler_file)
    print(metadata_file)

    print("\nRESULT FILES")
    print(loss_csv)
    print(loss_png)
    print(all_file)
    print(clean_file)
    print(outlier_file)

    print("\nDONE.")


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":
    main()
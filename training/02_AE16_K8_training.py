"""
02_AE16_K8_training.py

AE latent representation and K-means clustering used in the
human AD transcriptomic analysis.

Original workflow:
    MASTER_AE16_DAE.csv
        -> library-size normalization
        -> CPM
        -> log1p
        -> StandardScaler
        -> AE: 34 -> 24 -> 16 -> 24 -> 34
        -> 16-dimensional latent representation
        -> K-means clustering (K=8)

Outputs:
    results/AE16_training_loss.csv
    results/AE16_latent.csv
    results/AE16_K8_results.csv

    models/ae16_model.pt
    models/ae16_scaler.pkl
    models/ae16_kmeans_k8.pkl
    models/ae16_metadata.json
"""

from pathlib import Path
import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
LATENT_DIM = 16
HIDDEN_DIM = 24
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EPOCHS = 200
NOISE_STD = 0.03
N_CLUSTERS = 8

ROOT = Path(__file__).resolve().parent.parent

# Original prepared DAE-filtered master used for AE16.
INPUT_FILE = ROOT / "data" / "MASTER_AE16_DAE.csv"

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
# 2. AE16 MODEL
# Original architecture:
# 34 -> 24 -> 16 -> 24 -> 34
# ============================================================

class AE16(nn.Module):

    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, latent_dim)
        )

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, input_dim)
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z


# ============================================================
# 3. SAFE CSV READER
# ============================================================

def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:

        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                low_memory=False
            )

            print(
                f"[OK] CSV loaded: "
                f"encoding={enc}"
            )

            return df

        except UnicodeDecodeError:
            pass

    return pd.read_csv(
        path,
        low_memory=False
    )


# ============================================================
# 4. LOAD EXPRESSION MATRIX
# ============================================================

def load_expression_data(path):

    df = read_csv(path)

    expr_cols = [
        c for c in df.columns
        if str(c).upper().startswith("SRR")
    ]

    if len(expr_cols) != 34:
        raise ValueError(
            f"Expected 34 SRR expression columns, "
            f"found {len(expr_cols)}."
        )

    df[expr_cols] = df[expr_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    missing = int(
        df[expr_cols]
        .isna()
        .sum()
        .sum()
    )

    if missing > 0:
        raise ValueError(
            f"Expression matrix contains "
            f"{missing} missing/non-numeric values."
        )

    X = df[expr_cols].values.astype(
        np.float32
    )

    if np.any(X < 0):
        raise ValueError(
            "Negative expression values detected. "
            "CPM/log1p preprocessing requires "
            "non-negative expression values."
        )

    print("\nINPUT DATA")
    print("Genes:", df.shape[0])
    print("Total columns:", df.shape[1])
    print("Expression columns:", len(expr_cols))

    return df, expr_cols, X


# ============================================================
# 5. ORIGINAL AE16 PREPROCESSING
#
# library-size normalization
# -> CPM
# -> log1p
# -> StandardScaler
# ============================================================

def preprocess_expression(X):

    library_sizes = (
        X.sum(
            axis=0,
            keepdims=True
        )
        + 1e-8
    )

    X_cpm = (
        X
        / library_sizes
    ) * 1e6

    X_log = np.log1p(
        X_cpm
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_log
    ).astype(
        np.float32
    )

    if not np.isfinite(
        X_scaled
    ).all():

        raise ValueError(
            "NaN or infinite values detected "
            "after AE16 preprocessing."
        )

    print("\nPREPROCESSING")
    print("Library-size normalization: OK")
    print("CPM: OK")
    print("log1p: OK")
    print("StandardScaler: OK")
    print("AE input matrix:", X_scaled.shape)

    return X_scaled, scaler, library_sizes


# ============================================================
# 6. TRAIN AE16
#
# Original parameters:
# epochs = 200
# batch = 512
# lr = 1e-3
# training noise = 0.03
# ============================================================

def train_ae16(X_scaled):

    torch.manual_seed(SEED)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = AE16(
        input_dim=X_scaled.shape[1],
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.MSELoss()

    history = []

    print("\n" + "=" * 70)
    print("TRAINING AE16")
    print("=" * 70)

    print("Device:", DEVICE)
    print(
        "Architecture:",
        f"{X_scaled.shape[1]} -> "
        f"{HIDDEN_DIM} -> "
        f"{LATENT_DIM} -> "
        f"{HIDDEN_DIM} -> "
        f"{X_scaled.shape[1]}"
    )

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0

        for (batch,) in loader:

            clean_x = batch.to(
                DEVICE
            )

            noisy_x = (
                clean_x
                + NOISE_STD
                * torch.randn_like(
                    clean_x
                )
            )

            optimizer.zero_grad()

            reconstructed, _ = model(
                noisy_x
            )

            loss = loss_function(
                reconstructed,
                clean_x
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                float(loss.item())
                * clean_x.size(0)
            )

        epoch_loss /= len(X_scaled)

        history.append(
            epoch_loss
        )

        if (
            epoch == 0
            or (epoch + 1) % 25 == 0
            or epoch == EPOCHS - 1
        ):

            print(
                f"Epoch "
                f"{epoch + 1:3d}/{EPOCHS} | "
                f"loss={epoch_loss:.6f}"
            )

    return model, history


# ============================================================
# 7. GENERATE FINAL 16-D LATENT REPRESENTATION
# ============================================================

def generate_latent(
    model,
    X_scaled
):

    model.eval()

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).to(
        DEVICE
    )

    with torch.no_grad():

        reconstructed, latent = model(
            X_tensor
        )

    reconstructed = (
        reconstructed
        .cpu()
        .numpy()
    )

    latent = (
        latent
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    reconstruction_error = np.mean(
        np.square(
            X_scaled
            - reconstructed
        ),
        axis=1
    )

    print("\nLATENT REPRESENTATION")
    print("Latent matrix:", latent.shape)
    print(
        "Mean reconstruction error:",
        float(
            reconstruction_error.mean()
        )
    )

    return (
        latent,
        reconstruction_error
    )


# ============================================================
# 8. K-MEANS K=8
# Original:
# KMeans(
#     n_clusters=8,
#     random_state=42,
#     n_init="auto"
# )
# ============================================================

def run_kmeans(
    latent
):

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=SEED,
        n_init="auto"
    )

    clusters = (
        kmeans.fit_predict(
            latent
        )
        + 1
    )

    print("\nK-MEANS K=8")

    counts = (
        pd.Series(
            clusters
        )
        .value_counts()
        .sort_index()
    )

    for cluster_id, count in counts.items():

        print(
            f"Cluster {cluster_id}: "
            f"{count} genes"
        )

    return (
        kmeans,
        clusters
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

def save_results(
    df,
    latent,
    reconstruction_error,
    clusters,
    history
):

    loss_df = pd.DataFrame({
        "epoch":
            np.arange(
                1,
                EPOCHS + 1
            ),
        "loss":
            history
    })

    loss_file = (
        RESULT_DIR
        / "AE16_training_loss.csv"
    )

    loss_df.to_csv(
        loss_file,
        index=False
    )

    latent_df = pd.DataFrame(
        latent,
        columns=[
            f"z{i}"
            for i in range(
                1,
                LATENT_DIM + 1
            )
        ]
    )

    latent_file = (
        RESULT_DIR
        / "AE16_latent.csv"
    )

    latent_df.to_csv(
        latent_file,
        index=False
    )

    result = df.copy()

    for i in range(
        LATENT_DIM
    ):

        result[
            f"z{i + 1}"
        ] = latent[:, i]

    result[
        "cluster_kmeans_k8"
    ] = clusters

    result[
        "AE16_reconstruction_error"
    ] = reconstruction_error

    result_file = (
        RESULT_DIR
        / "AE16_K8_results.csv"
    )

    result.to_csv(
        result_file,
        index=False
    )

    return (
        loss_file,
        latent_file,
        result_file
    )


# ============================================================
# 10. SAVE PRETRAINED MODEL
# ============================================================

def save_model_files(
    model,
    scaler,
    kmeans,
    expr_cols,
    library_sizes,
    n_genes
):

    model_file = (
        MODEL_DIR
        / "ae16_model.pt"
    )

    scaler_file = (
        MODEL_DIR
        / "ae16_scaler.pkl"
    )

    kmeans_file = (
        MODEL_DIR
        / "ae16_kmeans_k8.pkl"
    )

    metadata_file = (
        MODEL_DIR
        / "ae16_metadata.json"
    )

    library_file = (
        MODEL_DIR
        / "ae16_training_library_sizes.npy"
    )

    torch.save(
        model.state_dict(),
        model_file
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    joblib.dump(
        kmeans,
        kmeans_file
    )

    np.save(
        library_file,
        library_sizes
    )

    metadata = {

        "model_name":
            "Human_AE16",

        "input_dataset":
            "MASTER_AE16_DAE.csv",

        "input_description":
            "Prepared DAE-filtered human gene-expression matrix",

        "n_input_genes":
            int(n_genes),

        "n_expression_features":
            int(len(expr_cols)),

        "expression_columns":
            [
                str(c)
                for c in expr_cols
            ],

        "architecture":
            "34-24-16-24-34",

        "input_dim":
            int(len(expr_cols)),

        "hidden_dim":
            HIDDEN_DIM,

        "latent_dim":
            LATENT_DIM,

        "activation":
            "ReLU",

        "noise_type":
            "Gaussian",

        "noise_std":
            NOISE_STD,

        "optimizer":
            "Adam",

        "learning_rate":
            LEARNING_RATE,

        "loss":
            "MSE",

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "seed":
            SEED,

        "preprocessing": [
            "library-size normalization",
            "CPM",
            "log1p",
            "StandardScaler"
        ],

        "clustering":
            "KMeans",

        "n_clusters":
            N_CLUSTERS,

        "kmeans_random_state":
            SEED,

        "kmeans_n_init":
            "auto"
    }

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    return (
        model_file,
        scaler_file,
        kmeans_file,
        metadata_file,
        library_file
    )


# ============================================================
# 11. VERIFY SAVED FILES
# ============================================================

def verify_saved_files(
    model_file,
    scaler_file,
    kmeans_file,
    input_dim
):

    test_model = AE16(
        input_dim=input_dim,
        latent_dim=LATENT_DIM
    )

    state_dict = torch.load(
        model_file,
        map_location="cpu"
    )

    test_model.load_state_dict(
        state_dict
    )

    test_model.eval()

    test_scaler = joblib.load(
        scaler_file
    )

    test_kmeans = joblib.load(
        kmeans_file
    )

    if (
        test_scaler.n_features_in_
        != input_dim
    ):

        raise ValueError(
            "Saved scaler dimension "
            "does not match AE16 input."
        )

    if (
        test_kmeans.n_clusters
        != N_CLUSTERS
    ):

        raise ValueError(
            "Saved K-means model "
            "does not contain K=8."
        )

    print(
        "[OK] AE16 model reloaded."
    )

    print(
        "[OK] AE16 scaler reloaded."
    )

    print(
        "[OK] K-means K=8 model reloaded."
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HUMAN AE16 + K-MEANS K=8 PIPELINE")
    print("=" * 70)

    print("Input:", INPUT_FILE)
    print("Repository:", ROOT)

    # --------------------------------------------------------
    # Load prepared DAE-filtered expression matrix
    # --------------------------------------------------------

    df, expr_cols, X = (
        load_expression_data(
            INPUT_FILE
        )
    )

    # --------------------------------------------------------
    # Original preprocessing
    # --------------------------------------------------------

    X_scaled, scaler, library_sizes = (
        preprocess_expression(
            X
        )
    )

    # --------------------------------------------------------
    # Train AE16
    # --------------------------------------------------------

    model, history = (
        train_ae16(
            X_scaled
        )
    )

    # --------------------------------------------------------
    # Generate z=16
    # --------------------------------------------------------

    latent, reconstruction_error = (
        generate_latent(
            model,
            X_scaled
        )
    )

    # --------------------------------------------------------
    # K-means K=8
    # --------------------------------------------------------

    kmeans, clusters = (
        run_kmeans(
            latent
        )
    )

    # --------------------------------------------------------
    # Save result tables
    # --------------------------------------------------------

    (
        loss_file,
        latent_file,
        result_file
    ) = save_results(
        df,
        latent,
        reconstruction_error,
        clusters,
        history
    )

    # --------------------------------------------------------
    # Save pretrained objects
    # --------------------------------------------------------

    (
        model_file,
        scaler_file,
        kmeans_file,
        metadata_file,
        library_file
    ) = save_model_files(
        model,
        scaler,
        kmeans,
        expr_cols,
        library_sizes,
        len(df)
    )

    # --------------------------------------------------------
    # Verify saved objects
    # --------------------------------------------------------

    verify_saved_files(
        model_file,
        scaler_file,
        kmeans_file,
        len(expr_cols)
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AE16 + K8 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nDATA")
    print("Genes:", len(df))
    print(
        "Expression columns:",
        len(expr_cols)
    )

    print("\nAE16")
    print(
        "Architecture: "
        "34 -> 24 -> 16 -> 24 -> 34"
    )
    print(
        "Latent dimension:",
        LATENT_DIM
    )
    print(
        "Training noise:",
        NOISE_STD
    )
    print(
        "Learning rate:",
        LEARNING_RATE
    )
    print(
        "Batch size:",
        BATCH_SIZE
    )
    print(
        "Epochs:",
        EPOCHS
    )

    print("\nCLUSTERING")
    print(
        "K-means K:",
        N_CLUSTERS
    )
    print(
        "Random state:",
        SEED
    )

    print("\nMODEL FILES")
    print(model_file)
    print(scaler_file)
    print(kmeans_file)
    print(metadata_file)
    print(library_file)

    print("\nRESULT FILES")
    print(loss_file)
    print(latent_file)
    print(result_file)

    print("\nDONE.")


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":
    main()
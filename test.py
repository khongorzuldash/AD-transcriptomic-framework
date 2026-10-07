import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "sample_data.csv"
MODEL_FILE = ROOT / "models" / "dae_model.pt"
SCALER_FILE = ROOT / "models" / "dae_scaler.pkl"
METADATA_FILE = ROOT / "models" / "dae_metadata.json"
OUTPUT_FILE = ROOT / "example_results" / "test_output.csv"

class DAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z

for file in [DATA_FILE, MODEL_FILE, SCALER_FILE, METADATA_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Required file not found: {file}")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

df = pd.read_csv(DATA_FILE)

expr_cols = metadata["expression_columns"]

missing_cols = [c for c in expr_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Sample data is missing {len(missing_cols)} required expression columns: {missing_cols[:5]}")

if len(expr_cols) != metadata["input_dim"]:
    raise ValueError(f"Metadata expects {metadata['input_dim']} expression columns but contains {len(expr_cols)} column names.")

X_raw = df[expr_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

if np.isnan(X_raw).any():
    raise ValueError("Sample data contains missing or non-numeric expression values.")

if np.any(X_raw < 0):
    raise ValueError("Negative expression values detected before log1p.")

X_log = np.log1p(X_raw)

scaler = joblib.load(SCALER_FILE)
X_scaled = scaler.transform(X_log).astype(np.float32)

model = DAE(input_dim=metadata["input_dim"], latent_dim=metadata["latent_dim"])
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    reconstructed, latent = model(X_tensor)
    reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

threshold = metadata["outlier_threshold"]
outlier = reconstruction_error >= threshold

result = df.copy()
result["DAE_reconstruction_error"] = reconstruction_error
result["DAE_outlier"] = outlier
result["DAE_z1"] = latent[:, 0].numpy()
result["DAE_z2"] = latent[:, 1].numpy()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

print("=" * 65)
print("PRETRAINED DAE TEST COMPLETED SUCCESSFULLY")
print("=" * 65)
print("Input file:", DATA_FILE)
print("Genes tested:", len(result))
print("Expression columns:", len(expr_cols))
print("Architecture:", metadata["architecture"])
print("Outlier threshold:", threshold)
print("Predicted clean:", int((~outlier).sum()))
print("Predicted outlier:", int(outlier.sum()))
print()
print("First five predictions:")
print(result[["DAE_reconstruction_error", "DAE_outlier", "DAE_z1", "DAE_z2"]].head().to_string(index=False))
print()
print("Output saved:", OUTPUT_FILE)
print("=" * 65)

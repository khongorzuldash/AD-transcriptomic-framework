# Deep Learning-Enhanced Transcriptomic Analysis of Alzheimer's Disease

This repository contains the custom computational code associated with the manuscript:

"Deep learning-enhanced RNA-seq reveals cryptic biologically coherent transcriptional signatures in Alzheimer's disease"

The repository provides scripts for denoising autoencoder (DAE)-based gene filtering, autoencoder (AE) latent representation, K-means clustering, gene-level machine learning, SHAP interpretation, and reproducible figure generation.

A minimal working example, including a small example expression dataset and pretrained DAE artifacts, is provided to allow immediate testing of the core DAE workflow without downloading or retraining the complete dataset.

## Analysis workflow

The principal computational workflow is:

```text
RNA-seq expression matrix
        ↓
DAE-based gene filtering
        ↓
AE latent representation (z = 16)
        ↓
K-means clustering (K = 8)
        ↓
Gene-level machine learning
        ↓
SHAP interpretation
        ↓
Figure generation
```

The DAE is used for reconstruction-based filtering of genes, whereas the downstream AE generates a 16-dimensional latent representation for clustering and subsequent analyses.

## Repository structure

```text
AD-transcriptomic-framework/
│
├── README.md
├── requirements.txt
├── .gitignore
├── test.py
│
├── data/
│   └── sample_data.csv
│
├── models/
│   ├── dae_model.pt
│   ├── dae_scaler.pkl
│   └── dae_metadata.json
│
├── training/
│   ├── 01_DAE_training.py
│   ├── 02_AE16_K8_training.py
│   ├── 03_gene_level_ML.py
│   ├── 04_SHAP_analysis.py
│   └── 05_figure_generation.py
│
├── example_results/
│   └── expected_output.csv
│
├── results/
└── figures/
```

## Requirements

Python 3.10 or later is recommended.

The principal Python dependencies are listed in `requirements.txt`.

Install the required packages using:

```bash
pip install -r requirements.txt
```

Major dependencies include:

- NumPy
- pandas
- scikit-learn
- PyTorch
- XGBoost
- SHAP
- matplotlib
- joblib

# Reproducibility modes

This repository supports two complementary reproducibility modes.

## 1. Minimal working example

A small example dataset and pretrained DAE artifacts are included directly in the repository.

This example is intended to verify the DAE inference pipeline without requiring the complete transcriptomic dataset or retraining the model.

The example dataset is:

```text
data/sample_data.csv
```

It contains 100 example genes and preserves the complete 34-sample expression structure required by the pretrained DAE.

The pretrained artifacts are:

```text
models/dae_model.pt
models/dae_scaler.pkl
models/dae_metadata.json
```

Run the example from the repository root:

```bash
python test.py
```

The script:

1. loads the example expression matrix;
2. identifies the 34 RNA-seq expression columns;
3. applies the stored preprocessing transformation;
4. loads the pretrained DAE;
5. calculates gene-level reconstruction errors;
6. applies the stored reconstruction-error threshold;
7. generates the two-dimensional DAE latent coordinates; and
8. writes the predictions to an output CSV file.

The generated output is:

```text
example_results/test_output.csv
```

A reference output is provided in:

```text
example_results/expected_output.csv
```

A successful run reports:

```text
PRETRAINED DAE TEST COMPLETED SUCCESSFULLY
```

The example is provided solely as a lightweight reproducibility test and is not intended to reproduce the full-scale biological results of the manuscript.

# Full analysis workflow

The scripts in `training/` document the major computational stages used for the full transcriptomic analysis.

Large full-scale expression matrices and intermediate analysis tables are not redistributed through this GitHub repository.

The complete analyses should be performed using the publicly available transcriptomic datasets described in the manuscript and Supplementary Information.

## Step 1 — DAE-based gene filtering

Script:

```text
training/01_DAE_training.py
```

Run:

```bash
python training/01_DAE_training.py
```

The DAE operates on the 34-sample gene-expression representation.

The architecture is:

```text
34 → 64 → 2 → 64 → 34
```

The model is trained using reconstruction loss with Gaussian noise introduced during training.

Gene-level reconstruction error is subsequently used for reconstruction-based filtering.

The script documents the preprocessing, model architecture, optimization procedure, reconstruction-error calculation, latent-coordinate extraction, and generation of clean/outlier gene tables.

For full-data training, the processed full expression matrix should be placed in the location specified by the script documentation.

## Step 2 — AE latent representation and K-means clustering

Script:

```text
training/02_AE16_K8_training.py
```

Run:

```bash
python training/02_AE16_K8_training.py
```

The downstream AE architecture is:

```text
34 → 24 → 16 → 24 → 34
```

The resulting latent representation therefore contains 16 dimensions.

Expression values are processed using the normalization and scaling procedures implemented in the script before AE training.

K-means clustering is subsequently performed in the complete 16-dimensional latent space using:

```text
K = 8
```

The two-dimensional displays generated for visualization do not replace the full 16-dimensional representation used for clustering.

## Step 3 — Gene-level machine learning

Script:

```text
training/03_gene_level_ML.py
```

Run:

```bash
python training/03_gene_level_ML.py
```

The final gene-level machine-learning input contains:

```text
34 expression features
+ 16 AE latent features
= 50 predictor features
```

The binary annotation used as the target variable is:

```text
is_protein_coding
```

where:

```text
1 = protein-coding
0 = non-protein-coding
```

The full-analysis intermediate table is referred to as:

```text
MASTER_AE16_AE_50.csv
```

This intermediate table is not required for the minimal working example and is not redistributed as part of the lightweight GitHub example.

The classifiers evaluated in the gene-level analysis are:

- XGBoost
- Random Forest
- K-nearest neighbors
- Decision Tree

Performance is evaluated using 10-fold cross-validation.

The XGBoost model is subsequently used for SHAP-based interpretation.

## Step 4 — SHAP analysis

Script:

```text
training/04_SHAP_analysis.py
```

Run:

```bash
python training/04_SHAP_analysis.py
```

SHAP analysis is performed using the final XGBoost model.

The script calculates SHAP values and global feature importance, including:

```text
mean absolute SHAP value
sum of SHAP values
sum of absolute SHAP values
```

The script also generates SHAP summary visualizations.

## Step 5 — Figure generation

Script:

```text
training/05_figure_generation.py
```

Run:

```bash
python training/05_figure_generation.py
```

The figure-generation script uses outputs from the preceding analysis stages to generate reproducible visualizations, including:

- DAE training-loss visualization
- AE training-loss visualization
- AE latent-space visualization
- gene-level machine-learning ROC curves
- global SHAP importance visualization

Figures are exported at high resolution for publication-quality output.

# Data availability

The transcriptomic data analyzed in the study were obtained from publicly available data resources.

Dataset accession numbers and the corresponding preprocessing and analysis descriptions are provided in the manuscript and Supplementary Information.

Large raw sequencing datasets and full-scale derived expression matrices are not duplicated in this GitHub repository.

Instead, this repository provides:

- the custom analysis scripts;
- the model architectures and analysis parameters;
- a lightweight example expression dataset;
- pretrained DAE artifacts; and
- an expected example output.

This design allows the core DAE inference workflow to be tested directly while avoiding unnecessary redistribution of large public transcriptomic datasets.

# Example data

The example file:

```text
data/sample_data.csv
```

contains a small subset of genes derived from the processed expression-data structure used by the DAE workflow.

The example retains all 34 expression features required by the pretrained model.

It is provided exclusively for testing the computational workflow and should not be interpreted as an independent biological dataset.

# Pretrained DAE

The repository includes the pretrained DAE artifacts required by the minimal working example:

```text
models/dae_model.pt
models/dae_scaler.pkl
models/dae_metadata.json
```

The model architecture is:

```text
34 → 64 → 2 → 64 → 34
```

The stored scaler and metadata are loaded by `test.py` so that the example data are processed consistently with the pretrained model.

# Expected output

Running:

```bash
python test.py
```

generates:

```text
example_results/test_output.csv
```

The repository also contains:

```text
example_results/expected_output.csv
```

for comparison with the expected example result.

The output includes:

```text
DAE_reconstruction_error
DAE_outlier
DAE_z1
DAE_z2
```

for each example gene.

# Reproducibility

Random seeds are fixed where applicable.

The repository documents the principal:

- preprocessing steps;
- neural-network architectures;
- latent dimensions;
- training parameters;
- clustering parameters;
- machine-learning models;
- cross-validation strategy;
- SHAP analysis; and
- figure-generation procedures.

The minimal example can be executed independently using the files distributed in this repository.

Full-scale analyses require the corresponding public transcriptomic datasets and the intermediate data preparation described in the manuscript and Supplementary Information.

# Citation

If you use this workflow, please cite the associated manuscript:

**Dashdondov Khongorzul et al.  
"Deep learning-enhanced RNA-seq reveals cryptic biologically coherent transcriptional signatures in Alzheimer's disease."**

Full bibliographic information will be added following publication.

# Contact

For questions regarding the computational workflow, please contact:

**Khongorzul Dashdondov**

Email: khongorzul.cbnu@gmail.com

# License

This repository is provided for academic and research use.
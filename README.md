# Deep learning-enhanced transcriptomic analysis of Alzheimer's disease

This repository contains the custom analysis code associated with the manuscript:

**"Deep learning-enhanced RNA-seq reveals cryptic biologically coherent transcriptional signatures in Alzheimer's disease"**

The repository provides code for denoising autoencoder (DAE)-based gene filtering, autoencoder latent representation, K-means clustering, gene-level machine learning, SHAP interpretation, and figure generation.

## Analysis workflow

RNA-seq expression data
→ DAE-based filtering
→ AE latent representation (z = 16)
→ K-means clustering (K = 8)
→ gene-level machine learning
→ SHAP interpretation

## Repository structure

```text
AD_GITHUB/
├── README.md
├── requirements.txt
├── test.py
├── data/
│   └── sample_data.csv
├── models/
│   ├── dae_model.pt
│   ├── dae_scaler.pkl
│   └── dae_metadata.json
├── training/
│   ├── 01_DAE_training.py
│   ├── 02_AE16_K8_training.py
│   ├── 03_gene_level_ML.py
│   ├── 04_SHAP_analysis.py
│   └── 05_figure_generation.py
├── example_results/
│   └── expected_output.csv
├── results/
└── figures/
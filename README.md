# Do Heterogeneous Graph Neural Networks Improve Cross-Cohort Glioma Grading?

**A TCGA-CGGA Benchmark with Graph Ablation and Explainability**

## Overview

This repository contains the code, data splits, and configuration files for the paper:

> **Do Heterogeneous Graph Neural Networks Improve Cross-Cohort Glioma Grading? A TCGA-CGGA Benchmark with Graph Ablation and Explainability**

We benchmark **7 heterogeneous GNN architectures** across **4 class-balancing pipelines** for binary glioma grade classification (LGG vs GBM), trained on **TCGA** (837 patients) and externally validated on **CGGA** (284 patients). We compare against **4 tabular baselines** and apply **5 graph-native XAI methods** alongside **SHAP-based tabular attribution**.

### Key Findings

- **MOGAT+CTGAN** achieved the highest TCGA test AUC (0.9203); **RGCN+CTGAN** achieved the highest CGGA AUC (0.7853)
- No GNN statistically outperformed the best tabular baseline (MLP, AUC = 0.9307)
- Gene-Patient bipartite edges are essential (−0.095 TCGA, −0.212 CGGA when removed)
- Patient-Patient topology encodes cohort-specific structure that can hinder cross-cohort transfer (+0.030 CGGA when removed)
- Both graph-native and SHAP methods converge on IDH1, IDH2, and age at diagnosis as dominant predictors

## Models Benchmarked

| Architecture | Type | Reference |
|---|---|---|
| HeteroGATv2 | Bidirectional GAT + GCN | Brody et al., 2021 |
| MOGAT | Dual-stream fusion (GAT + MLP) | Tanvir et al., 2024 |
| HyperTMO | Hypergraph convolution | Wang et al., 2024 |
| RGCN | Mean-pooling + Relational GCN | Schlichtkrull et al., 2018 |
| VEGN | Learned edge weights | Cheng et al., 2021 |
| FastHGTConv | Heterogeneous Graph Transformer | Hu et al., 2020 |
| SGNN | Mean-pooling + Chebyshev spectral | Defferrard et al., 2016 |

## Class-Balancing Pipelines

- **No Balancing** — inverse-frequency class weights only
- **SMOTE** — synthetic minority oversampling (k=3)
- **CTGAN** — conditional tabular GAN (150 epochs, batch 50, PAC 10)
- **ROS** — random oversampling

## Requirements

```
python==3.11
torch>=2.0
torch-geometric
numpy
pandas
scikit-learn
optuna
sdv
shap
xgboost
umap-learn
matplotlib
seaborn
scipy
```

### Installation

```bash
git clone https://github.com/[your-username]/glioma-gnn-benchmark.git
cd glioma-gnn-benchmark
pip install -r requirements.txt
```

## Usage

### Full Pipeline

```bash
python glioma_gnn_pipeline.py
```

This runs all 28 model-pipeline combinations, tabular baselines, ablation studies, graph strategy comparison, CTGAN quality assessment, and XAI analysis. Expected runtime: ~8-12 hours on an NVIDIA RTX 3050 (8 GB VRAM).

### Jupyter Notebook

```bash
jupyter notebook V8.ipynb
```

### Reproduce Results Only (from saved CSVs)

```python
import pandas as pd

results = pd.read_csv('results/results_final.csv')
print(results[results.Dataset == 'TCGA Test'].sort_values('AUC', ascending=False).head())
print(results[results.Dataset == 'CGGA'].sort_values('AUC', ascending=False).head())
```

## Datasets

| Dataset | Source | Patients | Grade 0 | Grade 1 |
|---|---|---|---|---|
| TCGA | [UCI Repository](https://doi.org/10.24432/C5R62J) | 837 | 485 | 352 |
| CGGA | [CGGA Portal](https://cgga-cns.org.cn/) | 284 | 182 | 102 |

Both datasets are publicly available. Patients under 18 were excluded.

## Experiment Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| HPO trials | 30 (Optuna TPE) |
| CV folds | 5 |
| Max epochs | 200 |
| Early stopping patience | 20 |
| Train/Test split | 80/20 (stratified) |
| Threshold calibration | G-mean, [0.20, 0.80], step 0.005 |
| Bootstrap resamples | 10,000 |

## Results Summary

### Best Configurations

| Dataset | Best GNN | AUC | Best Tabular | AUC |
|---|---|---|---|---|
| TCGA Test | MOGAT+CTGAN | 0.9203 | MLP | 0.9307 |
| CGGA | RGCN+CTGAN | 0.7853 | Random Forest | 0.7960 |

### Edge-Type Ablation (MOGAT+CTGAN)

| Configuration | TCGA AUC | ΔAUC | CGGA AUC | ΔAUC |
|---|---|---|---|---|
| Full graph | 0.9203 | — | 0.7306 | — |
| No Patient-Patient | 0.9133 | −0.007 | 0.7606 | +0.030 |
| No Gene-Gene | 0.9196 | −0.001 | 0.7421 | +0.012 |
| No Gene-Patient | 0.8249 | −0.095 | 0.5188 | −0.212 |

## License

This project is licensed under the MIT License.

## Acknowledgments

- Department of Computer Science, American International University - Bangladesh (AIUB)
- TCGA and CGGA consortia for making genomic data publicly available

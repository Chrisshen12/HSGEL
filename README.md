# HSGEL: Hot-Start Graph Ensemble Learning

This repository contains the official implementation of **HSGEL**, an efficient graph ensemble learning framework for large-scale graph learning.

> **Paper:** [Hot-Start Graph Ensemble Learning: Tackling Large Networks with Shared Initialization and Multi-Branch Diversification](#)  
> **Authors:** Jiajun Shen, Yufei Jin, Xingquan Zhu
---
The camera-ready paper for ICDM 2026 can be found at: 
## Overview

**HSGEL (Hot-Start Graph Ensemble Learning)** is an efficient framework for large-scale graph ensemble learning that generates diverse GNN experts from a **shared model**, avoiding the repeated cold-start training required by conventional ensemble methods.

HSGEL consists of three key components:

- **Hot-Start Graph Ensemble Learning:** A shared GNN is used to initialize multiple graph experts, enabling efficient ensemble generation without repeatedly training GNNs from scratch.

- **Branch-wise Sampling Diversification:** Mini-batch training uses different neighborhood sampling configurations across branches to capture complementary graph structures and multi-scale receptive fields.

- **Correlation-Penalized Residual (CPR) Weighting:** The best-performing GNN experts are adaptively combined using CPR weighting, which accounts for both prediction quality and model correlation.

- **Theoretical Diversity Guarantee:** A theoretical lower bound establishes the diversity enhancement achieved through multi-branch training with diversified graph samplers.

Overall, HSGEL provides an efficient approach to generating **accurate and diverse graph experts** while reducing the computational cost of large-scale GNN ensemble learning.

## Method

The overall HSGEL framework consists of three training stages:
![HSGEL Framework](HSGEL_framework_v1.pdf)

Initial Lifting: this stage trains a shared GNN using a fixed sampling configuration, as the hot-start for the next stage. 

Multi-branch Diversification: This stage enters multiple branch GNN training, each starting from the hot-start, using different sampler configurations for each branch, resulting in multiple branches with diverse experts. 

Correlation-penalized residual (CPR) weighting: This stages further minimize the risk of correlation between  selected branch experts to produce the final ensemble prediction.
## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

## Data preparation
* Homogeneous datasets for node classification
* Ogb datasets for node classification

These datasets include four medium-scale datasets. Please download them from pytorch geometric [pytorch-geometric-dataset](https://pytorch-geometric.readthedocs.io/en/2.5.2/modules/datasets.html#homogeneous-datasets).
You can download Ogb datasets [Ogbn](https://ogb.stanford.edu/docs/nodeprop/)

---
## Repository Structure

| File / Directory | Description |
|---|---|
| `train.py` | Main training script for HSGEL |
| `model.py` | GNN backbone implementations, including GCN and GraphSAGE |
| `dataset.py` | Dataset files and preprocessing utilities |
| `README.md` | Project documentation |

---

## Training

### 1. Train HSGEL

```bash
python train.py 
    --dataset reddit 
    --batch_size 0.1 
    --hop 2 
    --neighbors 10 
    --hop2 3 
    --neighbors2 5 
    --cycle 1
```
---
## Citation

If you find this work useful in your research, please cite:

```bibtex
```
If you encounter any issues, please feel free to reach out to me at jshen2024@fau.edu.

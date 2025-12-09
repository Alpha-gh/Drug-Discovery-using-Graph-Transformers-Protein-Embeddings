CMGT: Cross-Modal Graph Transformer for Drug–Target Binding Affinity Prediction

This repository contains the full implementation of CMGT (Cross-Modal Graph Transformer), a deep learning architecture for predicting drug–target binding affinity using:

  3D molecular structure features (atom-level graph + 3D coordinates)

  Protein sequence embeddings (ESM2)

  Cross-attention mechanism between ligand atoms and protein residues

  Geometric distance encoding using Fourier features

The project includes complete workflows for:

  Feature generation
  Model training (DAVIS + KIBA datasets)
  Evaluation (RMSE, Pearson, CI)
  Attention map extraction
  Visualization plots for research papers
  Comparison with baseline methods


Datasets
DAVIS Dataset

  Originally contains raw Kd values (0–10,000 nM).

  Converted to pKd = –log10(Kd in molar) for stable learning.

  Train/validation split: 80% / 20%

KIBA Dataset

  Already normalized affinity score.

  No log transform required.

  Train/validation split: 80% / 20%


Model Overview

CMGT consists of:

  1. Graph Transformer Encoder

    Processes ligand atom features + structure.

  2. Fourier Distance Encoding

    Encodes 3D geometric context using spectral features.

  3. Protein Transformer Projection

    Projects ESM2 embeddings into a common latent space.

  4. Cross-Attention Layers

    Two-way attention:

    Atom → Protein

    Protein → Atom

  5. Pooling + Regression Head

    Combines ligand + protein representations → predicts binding affinity.

  

# Neural Collaborative Filtering with PyTorch

A from-scratch PyTorch implementation of the full model family from [He et al., "Neural Collaborative Filtering" (WWW 2017)](https://dl.acm.org/doi/10.1145/3038912.3052569) — MF with bias terms, GMF, MLP, and NeuMF — plus a BPR learning-to-rank extension. This project demonstrates how to express classical collaborative filtering as a neural network and progressively add expressiveness by replacing dot products with learned non-linear interactions.

## What I built

### Matrix Factorization with bias terms
Extended a standard MF implementation to include per-user bias ($\mu_u$), per-item bias ($\mu_i$), and a global intercept ($\mu$):

$$\hat{R}_{ui} = \vec{P_u} \cdot \vec{Q_i} + \mu_u + \mu_i + \mu$$

In PyTorch, this is implemented as embedding layers for $P$, $Q$, user biases, and item biases, trained with MSE loss via SGD. Adding bias terms dropped RMSE from ~2.3 to ~2.0 — a significant improvement from a small architectural change.

### GMF — Generalized Matrix Factorization
Replaced the dot product with element-wise multiplication followed by a learned linear output layer, allowing the model to weight each latent dimension independently:

$$\hat{y}_{ui} = \sigma(h^T (p_u \odot q_i))$$

This makes the interaction function learnable rather than fixed, which is the key insight of the NCF framework.

### MLP — Multi-Layer Perceptron
Concatenated user and item embeddings and passed them through a fully-connected tower with ReLU activations, following the architecture specified in section 4.1 of the paper (layer sizes halved at each level). This architecture can capture arbitrary non-linear user-item interactions that neither dot products nor element-wise products can represent.

### NeuMF — Neural Matrix Factorization
Combined GMF and MLP in a single hybrid model with separate embedding matrices for each tower, concatenating their output vectors before a final prediction layer:

$$\hat{y}_{ui} = \sigma\left(h^T \begin{bmatrix} \phi^{GMF} \\ \phi^{MLP} \end{bmatrix}\right)$$

Separate embeddings allow GMF and MLP to learn complementary representations. This is the strongest pointwise model in the family.

### BPR — Learning to Rank
Implemented Bayesian Personalized Ranking on top of the NeuMF architecture. Rather than predicting ratings pointwise, BPR optimizes the probability that a user prefers an observed item over an unobserved one:

$$\text{BPR-OPT} = \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda \|\Theta\|^2$$

I implemented the pairwise triple sampling strategy — for each positive (user, item) pair in the training set, sampling a negative item the user hasn't rated — and the corresponding training loop.

### Evaluation
All 5 models evaluated on a held-out test set using RMSE (pointwise) and precision@K (ranking quality). NeuMF achieves the best RMSE; BPR achieves the best ranking precision.

## Results

| Model | RMSE | Notes |
|---|---|---|
| MF (no bias) | ~2.3 | baseline |
| MF + bias terms | ~2.0 | significant gain from small change |
| GMF | lower | learnable interaction weights |
| MLP | lower | non-linear interactions |
| NeuMF | best pointwise | GMF + MLP hybrid |
| BPR-NeuMF | best ranking precision | pairwise loss |

## Skills demonstrated

- Designing and training neural network architectures for recommendation in PyTorch (`nn.Module`, embedding layers, custom loss functions)
- Implementing research paper architectures from specification, including architectural details from multiple sections of the paper
- Building pairwise learning-to-rank training loops with negative sampling
- Evaluating models on both pointwise (RMSE) and ranking (precision@K) metrics
- Understanding how neural architectures progressively generalize classical CF

## Notebook

[`neural_collaborative_filtering.ipynb`](neural_collaborative_filtering.ipynb)

## Dependencies

```
torch, pandas, numpy, matplotlib
```

> PyTorch is pre-installed in Google Colab. For local execution: `pip install torch`.

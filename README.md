# Recommender Systems

This repository contains a series of projects I built to develop deep, hands-on expertise in recommender systems — from classical techniques to modern neural approaches. Each project involves implementing algorithms from their mathematical definitions, evaluating them rigorously on real data, and drawing analytical conclusions from the results. The dataset used throughout is [MovieLens](https://grouplens.org/datasets/movielens/).

Rather than using high-level libraries as black boxes, I focused on understanding the internals: writing gradient descent loops by hand, building custom evaluation metrics, implementing similarity functions, and designing experiments that replicate published research findings.

## Projects

### [Non-Personalized & Content-Based Filtering](non-personalized-content-based/)
Explores foundational recommendation techniques — popularity scoring, damped average ratings, demographic segmentation by gender and age, item association rules (co-occurrence and lift), and genre-based content filtering. Builds strong intuition for how data shape and sparsity affect recommendation quality before any personalization is applied.

### [Collaborative Filtering with Scikit-Surprise](collaborative-filtering/)
Implements a user-item average baseline, item-item nearest-neighbor collaborative filtering with a pre-computed truncated similarity model, and a genre-based content filter — all inside the scikit-surprise framework. Runs 5-fold cross-validation to compare RMSE across algorithms and demonstrates how subtracting the user-item baseline acts as a normalization step that improves downstream model performance.

### [Diversity-Aware Recommendations & Gradient Descent SVD](diversity-aware-recommendations/)
Replicates core findings from Ziegler et al.'s "Improving Recommendation Lists Through Topic Diversification" (WWW 2005). I implemented ILS (Intra-List Similarity) using the MovieLens tag genome, built Ziegler's greedy re-ranking diversification algorithm, and ran the paper's full experiment sweeping the diversification parameter `theta_f` across User-User and Item-Item base algorithms. I also implemented Funk SVD from scratch using stochastic gradient descent — including random initialization, per-rating gradient updates, and L2 regularization — and used it as an additional base algorithm in the replication.

### [Neural Collaborative Filtering with PyTorch](neural-collaborative-filtering/)
Implements the full model family from He et al.'s "Neural Collaborative Filtering" (WWW 2017) in PyTorch: Matrix Factorization with bias terms, GMF (element-wise product with learned weights), MLP (deep tower over concatenated embeddings), and NeuMF (hybrid GMF + MLP). Also extends the framework with a BPR learning-to-rank loss, implementing the pairwise sampling strategy from the BPR paper. All models are evaluated on RMSE and precision@K.

### [Adversarial Shilling Attacks](adversarial-shilling-attacks/)
Studies how injecting fake user profiles can manipulate recommender outputs. I implemented push and nuke attack variants using both average and bandwagon strategies, targeted specific items by exploiting their rating distributions, and measured attack effectiveness against KNN and SVD as victim models across varying attack sizes. The analysis quantifies the relative vulnerability of each algorithm and outlines practical mitigation strategies.

## Algorithms

| Algorithm | Family | Implementation | Project |
|---|---|---|---|
| Raw average rating | Non-personalized | From scratch | Non-personalized |
| Damped average rating | Non-personalized | From scratch | Non-personalized |
| Co-occurrence & lift-based association | Non-personalized | From scratch | Non-personalized |
| Genre cosine similarity | Content-based | From scratch | Non-personalized, CF |
| User-Item Average baseline | Memory-based CF | Custom Surprise algorithm | Collaborative filtering |
| Item-Item Collaborative Filtering | Memory-based CF | Custom Surprise algorithm | Collaborative filtering, Diversity |
| User-User Collaborative Filtering | Memory-based CF | Surprise | Diversity |
| Funk SVD (gradient descent) | Model-based CF | From scratch | Diversity |
| Matrix Factorization + bias terms | Neural / MF | PyTorch (`nn.Module`) | Neural CF |
| GMF — Generalized Matrix Factorization | Neural CF | PyTorch | Neural CF |
| MLP — Multi-Layer Perceptron | Neural CF | PyTorch | Neural CF |
| NeuMF — Neural Matrix Factorization | Neural CF | PyTorch | Neural CF |
| BPR — Bayesian Personalized Ranking | Learning to rank | PyTorch (custom loss + sampler) | Neural CF |
| Ziegler greedy re-ranking | Post-hoc diversification | From scratch | Diversity |
| Average shilling attack | Adversarial | From scratch | Shilling attacks |
| Bandwagon shilling attack | Adversarial | From scratch | Shilling attacks |

## Skills demonstrated

- Implementing recommendation algorithms directly from research papers and mathematical definitions
- Gradient descent optimization (Funk SVD, PyTorch training loops)
- PyTorch: custom `nn.Module` design, embedding layers, loss functions, training loops
- Offline evaluation: applied RMSE, precision@K, ILS, and lift across pointwise, ranking, and diversity problem types; used 5-fold cross-validation and hold-out protocols depending on the evaluation goal
- Understanding metric trade-offs: low RMSE does not imply good rankings, and high-precision lists are not necessarily diverse — multiple projects explicitly measure and analyze this tension
- Replicating published research results end-to-end
- Analyzing accuracy/diversity trade-offs and adversarial robustness
- Working with real-world sparse rating data and auxiliary datasets (tag genome)

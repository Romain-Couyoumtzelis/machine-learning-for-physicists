# Machine Learning for Physicists 📘🧪

Assignment-driven notebooks exploring practical machine learning techniques on physics-focused tasks. The repository is organized by homework modules for the 2024 session. Each module is self-contained and emphasizes reproducibility, baselines first, and clear evaluation.

Top-level contents:
- [HMW1 2024](HMW1%202024/)
- [HMW2 2024](HMW2%202024/)
- [HMW3 2024](HMW3%202024/)
- [README.md](README.md)

Note: This README summarizes the current notebooks and techniques and will be updated as new content is added.

---

## Notebooks overview 🔬

- HMW1 — Linear models and regularization
  - Notebook: [2024_assignment_1_romain_couyoumtzelis.ipynb](HMW1%202024/2024_assignment_1_romain_couyoumtzelis.ipynb)
  - Topics:
    - LASSO loss (L1) for sparse linear regression using a custom objective. See [Lasso regression (scikit-learn)](https://scikit-learn.org/stable/modules/linear_model.html#lasso).
    - Ridge regression (L2) estimator and closed-form solution. See [Ridge regression (scikit-learn)](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression).
    - Linear classification on a physics dataset with standard preprocessing and accuracy evaluation.
    - Custom optimization via [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
    - Data splits and transforms using [sklearn.model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) and [sklearn.preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html).

- HMW2 — Gaussian Mixture Models and Gibbs sampling
  - Notebook: [Assignment_2_Romain_Couyoumtzelis.ipynb](HMW2%202024/Assignment_2_Romain_Couyoumtzelis.ipynb)
  - Topics:
    - Gaussian Mixture Models (GMM) as generative models. See [GaussianMixture (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
    - Bayesian inference via Gibbs sampling to explore the posterior over cluster assignments and parameters. Background: [Gibbs sampling (Wikipedia)](https://en.wikipedia.org/wiki/Gibbs_sampling).
    - Label-switching resolution using the Hungarian algorithm ([scipy.optimize.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)).
    - Practical baselines and comparisons with scikit-learn’s EM-based GMM.
    - Iterative diagnostics and visualization with `matplotlib` and [tqdm](https://tqdm.github.io/).

- HMW3 — Quantum state tomography ⚛️
  - Notebook: [Assignment 3 - Quantum Tomography.ipynb](HMW3%202024/Assignment%203%20-%20Quantum%20Tomography.ipynb)
  - Data: [target_state_8.txt](HMW3%202024/target_state_8.txt), [target_state_12.txt](HMW3%202024/target_state_12.txt), [target_state_16.txt](HMW3%202024/target_state_16.txt)
  - Topics:
    - Quantum state tomography: reconstructing a density matrix ρ from measurement outcomes (Born rule) using neural networks.
    - Physical constraints in estimation: ρ is positive semidefinite (PSD) and trace-one; enforced via eigenvalue projection and normalization.
    - Estimation objectives: trained with KL divergence loss, optimized with [PyTorch](https://pytorch.org); includes log-transformation for numerical stability with small probabilities.
    - Validation metrics: state fidelity, predicted vs. ground truth scatter plots, and histograms for 8, 12, and 16-dimensional systems.
    - Numerical stability: regularization, dropout, and careful handling of eigen-decompositions for PSD constraints.

---

## Interesting techniques in the code ⚙️

- Sparse learning with L1 penalties (LASSO) to induce feature sparsity; contrasts with L2 (Ridge) which stabilizes coefficients without sparsity. [Lasso (scikit-learn)](https://scikit-learn.org/stable/modules/linear_model.html#lasso), [Ridge (scikit-learn)](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
- Closed-form ridge estimator and implementation details for numerical stability (conditioning with λI). [Ridge (scikit-learn)](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
- Custom objective optimization with [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) for maximum flexibility in loss design
- GMM clustering via both EM (through [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)) and Bayesian sampling (Gibbs) to compare point estimates vs posterior exploration
- Label permutation alignment using the Hungarian algorithm ([linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)) to evaluate inferred clusters against ground truth
- Quantum tomography with neural networks: KL divergence objectives, log-probability stabilization, PSD projection (eigendecomposition and eigenvalue clipping), and trace normalization

---

## Non-obvious libraries and choices for practitioners ✅

- [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html): direct optimization of custom losses, handy when deviating from standard estimators
- [scipy.optimize.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html): resolves label-switching in mixture models for fair evaluation
- [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html): EM-based baseline to contrast with sampling-based inference
- [tqdm](https://tqdm.github.io/): lightweight progress visualization for iterative sampling/training loops
- [PyTorch](https://pytorch.org): neural network training and optimization for quantum tomography objectives

Other common dependencies (as seen in imports):
- [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [Matplotlib](https://matplotlib.org), [scikit-learn](https://scikit-learn.org), [SciPy](https://scipy.org)

Fonts
- No custom fonts are referenced.

---

## Project structure 📁

```text
/
├─ README.md
├─ HMW1 2024/
├─ HMW2 2024/
└─ HMW3 2024/
```

Directory notes:
- HMW1 2024/: Linear models, regularization, and a classification task with evaluation.
- HMW2 2024/: GMMs, Gibbs sampling, and cluster alignment methodology.
- HMW3 2024/: Quantum state tomography; includes a notebook and target state files (8, 12, 16) for training and validation across system sizes.

---

If you want, we can extend each module with a short local README, standardize plotting/output directories (e.g., results/ or figures/), and pin the environment (requirements.txt or environment.yml). 🔧
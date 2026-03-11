# Data README

## Dataset Description

This project uses **synthetically generated datasets** only. No external data files need to be downloaded.

### Dataset: Covariance-Structured Gaussian Distributions

**Problem type:** Binary classification on probability distributions.

**Generation:** All datasets are generated programmatically in each notebook using `numpy.random`. The generation code is self-contained and seeded (seed=42) for reproducibility.

**Structure:**
- Each training example is a **probability distribution**, represented as a set of 2D samples drawn from a multivariate Gaussian.
- **Class +1 (horizontal):** Distributions with covariance `[[1.8, 0.05], [0.05, 0.25]]` (elongated horizontally). Means drawn from N(0, 0.6*I).
- **Class -1 (vertical):** Distributions with covariance `[[0.25, 0.05], [0.05, 1.8]]` (elongated vertically). Means drawn from N(0, 0.6*I).

**Key property:** The class means fully overlap in 2D space. A standard SVM trained on means performs at chance (~50%). The SMM captures covariance information through the expected RBF kernel (Equation 4 of the paper), achieving ~89% accuracy.

**Default parameters:**
- `N_PER_CLASS = 100` distributions per class (200 total)
- `N_SAMPLES = 20` samples drawn per distribution
- `D = 2` feature dimensions

## How to use

The dataset is generated inline in each notebook. No manual steps are required. The relevant function is `generate_cov_dataset()` defined at the top of each notebook, which calls `numpy.random.multivariate_normal`.

## Comparison to original paper dataset (Section 6.1)

The paper's synthetic dataset uses:
- 10-dimensional distributions
- 500 distributions per class (1000 total)
- Means drawn from N(m+=(1,...,1), 0.5*I₁₀) and N(m−=(2,...,2), 0.5*I₁₀)
- Covariances from Wishart distributions

Our dataset is simplified to 2D for visualization and speed, but is designed to more clearly demonstrate the core SMM advantage: capturing covariance-level distributional information that a mean-only SVM cannot.

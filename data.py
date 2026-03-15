from __future__ import annotations
import numpy as np

# synthetic_datasets
def make_synthetic_datasets(rng: np.random.Generator, n: int = 30_000, d: int = 50, k_true: int = 8):
    from sklearn.datasets import make_blobs, make_circles, make_moons

    datasets = {}

    X1, y1 = make_blobs(n_samples=n, centers=k_true, n_features=d, cluster_std=2.0, random_state=123)
    datasets["blobs"] = (X1.astype(np.float32), y1.astype(np.int32))

    X2, y2 = make_blobs(n_samples=n, centers=k_true, n_features=2, cluster_std=1.5, random_state=321)
    A = np.array([[0.6, -0.8], [0.4, 0.9]])
    X2 = X2 @ A
    if d > 2:
        X2 = np.hstack([X2, rng.normal(0, 0.1, size=(n, d - 2))])
    datasets["anisotropic"] = (X2.astype(np.float32), y2.astype(np.int32))

    X3, y3 = make_blobs(n_samples=n, centers=k_true, n_features=min(10, d), cluster_std=2.5, random_state=222)
    if d > X3.shape[1]:
        noise = rng.normal(0, 1.0, size=(n, d - X3.shape[1]))
        mask = rng.random(noise.shape) < 0.85
        noise[mask] *= 0.05
        X3 = np.hstack([X3, noise])
    datasets["high_dim_sparseish"] = (X3.astype(np.float32), y3.astype(np.int32))

    return datasets

# real_datasets
def make_real_datasets(rng: np.random.Generator):
    from sklearn.datasets import load_iris, fetch_covtype, fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    datasets = {}
    scaler = StandardScaler()

    iris = load_iris()
    X_iris = scaler.fit_transform(iris.data).astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    datasets["real_iris"] = (X_iris, y_iris)

    cov = fetch_covtype()
    X_cov = cov.data.astype(np.float32)
    y_cov = cov.target.astype(np.int32) - 1

    idx = rng.choice(len(X_cov), size=min(50_000, len(X_cov)), replace=False)
    X_cov = scaler.fit_transform(X_cov[idx]).astype(np.float32)
    y_cov = y_cov[idx]
    datasets["real_covertype"] = (X_cov, y_cov)

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_mnist = mnist.data.astype(np.float32)
    y_mnist = mnist.target.astype(np.int32)

    idx = rng.choice(len(X_mnist), size=min(20_000, len(X_mnist)), replace=False)
    X_mnist = X_mnist[idx]
    y_mnist = y_mnist[idx]

    X_mnist = PCA(n_components=50, random_state=42).fit_transform(X_mnist)
    X_mnist = scaler.fit_transform(X_mnist).astype(np.float32)
    datasets["real_mnist_pca50"] = (X_mnist, y_mnist)

    return datasets
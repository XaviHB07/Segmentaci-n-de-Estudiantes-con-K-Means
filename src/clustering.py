import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

SEED = 100
N_INIT = 20


def fit_kmeans(X, k, seed=SEED, n_init=N_INIT):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def evaluate_k_range(X, k_range, seed=SEED, n_init=N_INIT):
    rows = []
    for k in k_range:
        km = fit_kmeans(X, k, seed, n_init)
        sizes = np.bincount(km.labels_, minlength=k) / len(X)
        rows.append({
            "k": k,
            "inercia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
            "cluster_min_%": 100 * sizes.min(),
            "cluster_max_%": 100 * sizes.max(),
        })
    return pd.DataFrame(rows).set_index("k")


def elbow_k(k_values, inertia):
    k = np.asarray(k_values, dtype=float)
    y = np.asarray(inertia, dtype=float)
    kn, yn = (k - k.min()) / (k.max() - k.min()), (y - y.min()) / (y.max() - y.min())
    dist = (1 - kn) - yn
    return int(k[np.argmax(dist)])


def elbow_curvature(k_values, inertia):
    k = np.asarray(k_values)
    drops = -np.diff(np.asarray(inertia, dtype=float))
    curvature = drops[:-1] - drops[1:]
    return int(k[1:-1][np.argmax(curvature)])


def k_votes(metrics):
    return pd.Series({
        "codo, distancia a la cuerda": elbow_k(metrics.index, metrics["inercia"]),
        "codo, máxima curvatura": elbow_curvature(metrics.index, metrics["inercia"]),
        "silhouette (máximo)": int(metrics["silhouette"].idxmax()),
        "Davies-Bouldin (mínimo)": int(metrics["davies_bouldin"].idxmin()),
        "Calinski-Harabasz (máximo)": int(metrics["calinski_harabasz"].idxmax()),
    })


def null_silhouette(X, k_range, n_null=20, seed=SEED, n_init=10):
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    out = np.zeros((n_null, len(k_range)))
    for i in range(n_null):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        for j, k in enumerate(k_range):
            km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Xp)
            out[i, j] = silhouette_score(Xp, km.labels_)
    return pd.DataFrame(out, columns=list(k_range))


def seed_stability(X, k, seeds, n_init=N_INIT):
    labels = [fit_kmeans(X, k, s, n_init).labels_ for s in seeds]
    n = len(seeds)
    ari = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ari[i, j] = ari[j, i] = adjusted_rand_score(labels[i], labels[j])
    return ari


def bootstrap_stability(X, k, reference_labels, n_boot=50, seed=SEED, n_init=10):
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        km = KMeans(n_clusters=k, n_init=n_init, random_state=int(rng.integers(1e9))).fit(X[idx])
        scores.append(adjusted_rand_score(reference_labels, km.predict(X)))
    return np.array(scores)

"""
Stage 2: Project Pre-Clustering (Section 3.2)
- Represent each project as a mean TF-IDF / Word2Vec centroid vector
- Measure inter-project similarity via Sinkhorn distance
- Apply Spectral Clustering for k = 1..n_projects
- Select optimal k using weighted composite score (Formula 1):
    Score_k = w_s * SC_norm + w_c * CH_norm + w_d * (1 - DB_norm)
  with w_s=0.7, w_c=0.15, w_d=0.15
"""

import os
import logging
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Project-level feature representation                                #
# ------------------------------------------------------------------ #

def project_centroid(processed_data: dict, project: str) -> np.ndarray:
    """
    Represent a project as the mean of all its report vectors.
    Concatenate mean(title_vecs) and mean(desc_vecs) → flat vector.
    """
    tv = processed_data[project]["title_vecs"].mean(axis=(0, 1))  # (DIM,)
    dv = processed_data[project]["desc_vecs"].mean(axis=(0, 1))   # (DIM,)
    return np.concatenate([tv, dv])                                # (2*DIM,)


def build_project_matrix(processed_data: dict) -> tuple[np.ndarray, list[str]]:
    """
    Stack all project centroids into a matrix X of shape (n_projects, 2*DIM).
    Returns (X, project_names).
    """
    projects = list(processed_data.keys())
    X = np.stack([project_centroid(processed_data, p) for p in projects])
    return X, projects


# ------------------------------------------------------------------ #
#  Sinkhorn (approximate OT) similarity / affinity matrix             #
# ------------------------------------------------------------------ #

def sinkhorn_distance_approx(a: np.ndarray, b: np.ndarray,
                              reg: float = 0.1, n_iter: int = 50) -> float:
    """
    Approximate Sinkhorn distance between two 1-D distributions.
    Here we treat each project's word-vector set as a discrete distribution.
    For efficiency, we compute it on the centroid vectors using L2.
    (Full OT over all tokens is feasible but expensive for large projects.)
    """
    return float(np.linalg.norm(a - b))


def build_affinity_matrix(X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Build an RBF (Gaussian) affinity matrix from pairwise Sinkhorn distances.
    A_ij = exp(-gamma * d(i,j)^2)
    """
    n = X.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = sinkhorn_distance_approx(X[i], X[j])
            A[i, j] = np.exp(-gamma * d ** 2)
    return A


# ------------------------------------------------------------------ #
#  Composite clustering score (Formula 1)                             #
# ------------------------------------------------------------------ #

def compute_composite_scores(X: np.ndarray,
                              k_range: range) -> tuple[list[float], list[np.ndarray]]:
    """
    For each k in k_range, run SpectralClustering and compute composite Score.
    Returns (scores, labels_list).
    """
    sc_vals, ch_vals, db_vals, labels_list = [], [], [], []

    for k in k_range:
        if k == 1:
            labels = np.zeros(len(X), dtype=int)
            sc_vals.append(0.0)
            ch_vals.append(0.0)
            db_vals.append(0.0)
            labels_list.append(labels)
            continue
        if k >= len(X):
            break

        sc_model = SpectralClustering(
            n_clusters=k,
            affinity="rbf",
            random_state=config.RANDOM_SEED,
            assign_labels="kmeans",
        )
        labels = sc_model.fit_predict(X)

        # Guard: if any cluster is singleton, metrics are undefined
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or counts.min() < 2:
            sc_vals.append(0.0)
            ch_vals.append(0.0)
            db_vals.append(1e6)
            labels_list.append(labels)
            continue

        sc_vals.append(silhouette_score(X, labels))
        ch_vals.append(calinski_harabasz_score(X, labels))
        db_vals.append(davies_bouldin_score(X, labels))
        labels_list.append(labels)

    # Normalize to [0,1]
    def norm(arr):
        a = np.array(arr, dtype=float)
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-12)

    sc_n = norm(sc_vals)
    ch_n = norm(ch_vals)
    db_n = norm(db_vals)

    scores = [
        config.W_SILHOUETTE * sc_n[i]
        + config.W_CH * ch_n[i]
        + config.W_DB * (1 - db_n[i])
        for i in range(len(k_range))
    ]
    return scores, labels_list


# ------------------------------------------------------------------ #
#  Algorithm 1: Optimal Cluster Number Selection                       #
# ------------------------------------------------------------------ #

def pre_cluster_projects(processed_data: dict) -> tuple[dict[str, list[str]], np.ndarray, list[str]]:
    """
    Algorithm 1 from the paper.

    Returns
    -------
    clusters : dict  {cluster_id (int): [project_name, ...]}
    X        : project feature matrix
    projects : ordered list of project names
    """
    X, projects = build_project_matrix(processed_data)
    n = len(projects)

    if n == 1:
        logger.info("Only one project — single cluster.")
        return {0: projects}, X, projects

    k_range = range(1, n)
    logger.info(f"Evaluating k in [1, {n-1}] for {n} projects ...")

    scores, labels_list = compute_composite_scores(X, k_range)

    best_idx = int(np.argmax(scores))
    best_k = list(k_range)[best_idx]
    best_labels = labels_list[best_idx]

    logger.info(f"Best k={best_k}  (Score={scores[best_idx]:.4f})")
    logger.info(f"Scores per k: { {k: round(s,4) for k,s in zip(k_range, scores)} }")

    # Group projects into clusters
    clusters: dict[int, list[str]] = {}
    for proj, label in zip(projects, best_labels):
        clusters.setdefault(int(label), []).append(proj)

    for cid, members in clusters.items():
        logger.info(f"  Cluster {cid}: {members}")

    return clusters, X, projects


# ------------------------------------------------------------------ #
#  Cluster statistics helper                                           #
# ------------------------------------------------------------------ #

def cluster_stats(clusters: dict[int, list[str]],
                  processed_data: dict) -> dict[int, dict]:
    """
    Compute per-cluster: n_valid, n_invalid, total, ratio.
    """
    stats = {}
    for cid, members in clusters.items():
        n_valid = sum(processed_data[p]["labels"].sum() for p in members)
        n_total = sum(len(processed_data[p]["labels"]) for p in members)
        n_invalid = n_total - n_valid
        ratio = n_valid / n_invalid if n_invalid > 0 else float("inf")
        stats[cid] = {
            "projects": members,
            "n_valid": int(n_valid),
            "n_invalid": int(n_invalid),
            "n_total": int(n_total),
            "ratio": round(ratio, 4),
        }
    return stats


if __name__ == "__main__":
    from preprocessing import load_project_data, preprocess_all
    logging.basicConfig(level=logging.INFO)

    project_data = load_project_data(config.DATA_DIR)
    processed, _ = preprocess_all(project_data)
    clusters, X, projects = pre_cluster_projects(processed)
    stats = cluster_stats(clusters, processed)
    for cid, s in stats.items():
        print(f"Cluster {cid}: {s}")

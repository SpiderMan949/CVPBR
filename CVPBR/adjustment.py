"""
Stage 3: Cluster Adjustment (Section 3.3)

Algorithm 2 – Global Standard Deviation Minimization
  Constraints:
    - intra-cluster total samples >= N  (default 3000)
    - positive-to-negative ratio in [R_min, R_max]  (default [1/1.6, 1.6])

Algorithm 3 – Interference Ratio Based Adjustment
  Extra step: compute Interference Ratio (Formula 2) per project;
  projects below threshold α=0.05 are bound together and
  moved as a unit through Algorithm 2.
"""

import os
import logging
import itertools
from collections import Counter, defaultdict

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _cluster_counts(clusters: dict, processed_data: dict) -> dict:
    """Return {cid: (n_valid, n_invalid)} for current cluster assignment."""
    counts = {}
    for cid, members in clusters.items():
        nv = int(sum(processed_data[p]["labels"].sum() for p in members))
        ni = int(sum((processed_data[p]["labels"] == 0).sum() for p in members))
        counts[cid] = (nv, ni)
    return counts


def _global_std(counts: dict) -> float:
    """Global std of (n_invalid - n_valid) across all clusters."""
    diffs = [ni - nv for (nv, ni) in counts.values()]
    return float(np.std(diffs))


def _ratio_ok(nv: int, ni: int) -> bool:
    """Check if positive-to-negative ratio is in [R_min, R_max]."""
    if ni == 0:
        return False
    r = nv / ni
    return config.R_MIN <= r <= config.R_MAX


def _constraints_satisfied(clusters: dict, processed_data: dict) -> bool:
    """True iff every cluster satisfies size and ratio constraints."""
    counts = _cluster_counts(clusters, processed_data)
    for cid, (nv, ni) in counts.items():
        total = nv + ni
        if total < config.MIN_CLUSTER_SIZE:
            return False
        if not _ratio_ok(nv, ni):
            return False
    return True


# ------------------------------------------------------------------ #
#  Algorithm 2: Global Standard Deviation Minimization                 #
# ------------------------------------------------------------------ #

def adjustment_strategy_1(clusters: dict[int, list[str]],
                           processed_data: dict,
                           movable_projects: list[str] = None
                           ) -> dict[int, list[str]]:
    """
    Algorithm 2: iteratively reassign imbalanced projects to minimise
    global std of (n_invalid - n_valid) across clusters, subject to
    min sample size N and ratio [R_min, R_max].

    Parameters
    ----------
    clusters         : initial clustering {cid: [project, ...]}
    processed_data   : preprocessed data per project
    movable_projects : if given, only these are candidates for reassignment
                       (used by Algorithm 3 for bound groups)
    """
    import copy
    clusters = copy.deepcopy(clusters)

    # Step 1: identify imbalanced projects (more invalid than valid)
    extraction_set: list[str] = []
    for cid, members in list(clusters.items()):
        for p in list(members):
            nv = int(processed_data[p]["labels"].sum())
            ni = int((processed_data[p]["labels"] == 0).sum())
            if ni > nv:   # more invalid — bias candidate
                extraction_set.append(p)

    # If caller restricts movable projects, intersect
    if movable_projects is not None:
        extraction_set = [p for p in extraction_set if p in movable_projects]

    # Remove extracted projects from their current clusters
    for p in extraction_set:
        for cid in list(clusters.keys()):
            if p in clusters[cid]:
                clusters[cid].remove(p)
        # Drop empty clusters
    clusters = {cid: m for cid, m in clusters.items() if m}

    logger.info(f"[S1] Extracted {len(extraction_set)} imbalanced projects: {extraction_set}")

    # Step 2: iterative greedy reassignment
    max_iter = len(extraction_set) * len(clusters) * 10
    iteration = 0
    while extraction_set and iteration < max_iter:
        iteration += 1
        counts = _cluster_counts(clusters, processed_data)
        best_std = float("inf")
        best_proj = None
        best_cid = None

        for p in extraction_set:
            for cid in clusters:
                # Temporarily add p to cid
                nv_p = int(processed_data[p]["labels"].sum())
                ni_p = int((processed_data[p]["labels"] == 0).sum())
                old_nv, old_ni = counts[cid]
                new_counts = dict(counts)
                new_counts[cid] = (old_nv + nv_p, old_ni + ni_p)
                std = _global_std(new_counts)
                if std < best_std:
                    best_std = std
                    best_proj = p
                    best_cid = cid

        if best_proj is None:
            break

        # Apply best reassignment
        clusters[best_cid].append(best_proj)
        extraction_set.remove(best_proj)
        logger.debug(f"  Moved [{best_proj}] → Cluster {best_cid}  (std={best_std:.2f})")

        if _constraints_satisfied(clusters, processed_data):
            logger.info("[S1] All constraints satisfied early.")
            break

    # Reassign any remaining projects to the cluster with lowest std impact
    for p in extraction_set:
        counts = _cluster_counts(clusters, processed_data)
        nv_p = int(processed_data[p]["labels"].sum())
        ni_p = int((processed_data[p]["labels"] == 0).sum())
        best_cid = min(
            clusters.keys(),
            key=lambda c: _global_std({
                **counts,
                c: (counts[c][0] + nv_p, counts[c][1] + ni_p)
            })
        )
        clusters[best_cid].append(p)
        logger.info(f"  Forced [{p}] → Cluster {best_cid} (leftover)")

    logger.info("[S1] Adjustment complete.")
    for cid, members in clusters.items():
        counts = _cluster_counts(clusters, processed_data)
        nv, ni = counts[cid]
        logger.info(f"  Cluster {cid}: {members}  valid={nv} invalid={ni} "
                    f"total={nv+ni} ratio={nv/max(ni,1):.2f}")
    return clusters


# ------------------------------------------------------------------ #
#  Interference Ratio (Formula 2)                                      #
# ------------------------------------------------------------------ #

def _top_words(token_lists: list[list[str]], top_pct: float = config.TOP_WORD_PERCENT) -> set[str]:
    """Return the top `top_pct` fraction of words by frequency."""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    n_top = max(1, int(len(counter) * top_pct))
    return {w for w, _ in counter.most_common(n_top)}


def interference_ratio(cluster_members: list[str], processed_data: dict) -> dict[str, float]:
    """
    Compute per-project Interference Ratio (Formula 2) for every project
    in a cluster.

    Formula:
        IR = |W_all| / (|I_all| + |V_all| - |W_all|)
    where
        I_all = union of top-5% invalid-report words across all projects
        V_all = union of top-5% valid-report words across all projects
        W_all = I_all ∩ V_all

    We compute this for each project by treating it as a "leave-one-in"
    contribution: its own interference = |W_p| / (|I_p| + |V_p| - |W_p|)
    where I_p / V_p are its own top words.
    """
    ratios = {}
    for p in cluster_members:
        data = processed_data[p]
        invalid_mask = data["labels"] == 0
        valid_mask = data["labels"] == 1

        invalid_tokens = [t for i, tl in enumerate(data["desc_tokens"]) if invalid_mask[i] for t in tl]
        valid_tokens   = [t for i, tl in enumerate(data["desc_tokens"]) if valid_mask[i]  for t in tl]

        I_p = _top_words([invalid_tokens])
        V_p = _top_words([valid_tokens])
        W_p = I_p & V_p
        denom = len(I_p) + len(V_p) - len(W_p)
        ratios[p] = len(W_p) / denom if denom > 0 else 0.0

    return ratios


# ------------------------------------------------------------------ #
#  Algorithm 3: Interference Ratio Based Adjustment                    #
# ------------------------------------------------------------------ #

def adjustment_strategy_2(clusters: dict[int, list[str]],
                           processed_data: dict) -> dict[int, list[str]]:
    """
    Algorithm 3: interference-ratio-aware version of Algorithm 2.

    1. For each cluster, compute interference ratio per project.
    2. Projects with IR < α are "bound" together.
    3. Bound groups are moved collectively through Algorithm 2.
    """
    import copy
    clusters = copy.deepcopy(clusters)

    # Step 1: collect all bound groups across clusters
    bound_groups: list[list[str]] = []   # each group is a list of project names
    binding_list: list[str] = []         # flat list of all bound projects

    for cid, members in clusters.items():
        if len(members) < 2:
            continue
        ratios = interference_ratio(members, processed_data)
        low_ir = [p for p, r in ratios.items() if r < config.INTERFERENCE_ALPHA]

        logger.info(f"Cluster {cid} IR: { {p: round(r,4) for p,r in ratios.items()} }")
        logger.info(f"  Bound projects (IR < {config.INTERFERENCE_ALPHA}): {low_ir}")

        if len(low_ir) >= 2:
            bound_groups.append(low_ir)
            binding_list.extend(low_ir)

    if not bound_groups:
        logger.info("[S2] No bound groups found; falling back to Algorithm 2.")
        return adjustment_strategy_1(clusters, processed_data)

    # Step 2: apply Algorithm 2 with bound groups as movable units
    # Treat each bound group as a single movable "super-project"
    # by delegating to S1 with those projects marked movable
    all_bound = list(set(binding_list))
    adjusted = adjustment_strategy_1(clusters, processed_data, movable_projects=all_bound)
    return adjusted


# ------------------------------------------------------------------ #
#  Public entry point                                                  #
# ------------------------------------------------------------------ #

def adjust_clusters(clusters: dict[int, list[str]],
                    processed_data: dict,
                    strategy: str = "S3") -> dict[int, list[str]]:
    """
    strategy : "S1" | "S2" | "S3"
      S1 = Algorithm 2 only
      S2 = Algorithm 2 only (alias, for ablation study)
      S3 = Algorithm 3 (CVCBR full method)
    """
    logger.info(f"Running cluster adjustment with strategy={strategy}")
    if strategy in ("S1", "S2"):
        return adjustment_strategy_1(clusters, processed_data)
    elif strategy == "S3":
        return adjustment_strategy_2(clusters, processed_data)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    from preprocessing import load_project_data, preprocess_all
    from clustering import pre_cluster_projects
    logging.basicConfig(level=logging.INFO)

    project_data = load_project_data(config.DATA_DIR)
    processed, _ = preprocess_all(project_data)
    clusters, _, _ = pre_cluster_projects(processed)
    adjusted = adjust_clusters(clusters, processed, strategy="S3")
    print("Final clusters:", {k: v for k, v in adjusted.items()})

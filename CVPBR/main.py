"""
CVCBR — Main Pipeline
======================
Experimental protocol (Section 4):

1. Time-aware train/test split
   - Bug reports sorted chronologically per project
   - Split into 10 equal segments; first 9 = train, last 1 = test
   - ALL subsequent steps (clustering, adjustment, training) use TRAIN only
   - Test set is held out until final evaluation

2. Multi-seed repetition
   - Repeat full experiment with 5 different random seeds
   - Report mean (± std) across seeds for every metric

3. Statistical analysis
   - Wilcoxon signed-rank test  (CVCBR vs each baseline)
   - Cliff's δ effect size
   - 95% bootstrap confidence intervals for key metrics

Usage
-----
    python main.py [--strategy S3] [--data_dir ./data]
                   [--seeds 42,0,1,2,3] [--n_segments 10]
                   [--train_segments 9]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

import torch
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, precision_score, recall_score)
from sklearn.utils import resample

import config
from preprocessing import load_project_data, preprocess_all
from clustering import pre_cluster_projects, cluster_stats
from adjustment import adjust_clusters
from model import TextCNN, BugReportDataset

os.makedirs(config.MODEL_DIR,  exist_ok=True)
os.makedirs(config.RESULT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.RESULT_DIR, "run.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ======================================================================
#  CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CVCBR full pipeline")
    parser.add_argument("--strategy", choices=["S1", "S2", "S3"], default="S3")
    parser.add_argument("--data_dir", default=config.DATA_DIR)
    parser.add_argument("--seeds", default="42,0,1,2,3",
                        help="Comma-separated random seeds (default: 42,0,1,2,3)")
    parser.add_argument("--n_segments", type=int, default=10,
                        help="Total time segments (default: 10)")
    parser.add_argument("--train_segments", type=int, default=9,
                        help="Segments used for training (default: 9, last 1 = test)")
    parser.add_argument("--time_col", default=None,
                        help="Column name for timestamp (auto-detected if None)")
    return parser.parse_args()


# ======================================================================
#  Time-aware train/test split  (Section 4: time-aware setting)
# ======================================================================

_TIME_COL_ALIASES = {
    "created_at", "created", "date", "timestamp", "time",
    "open_date", "submission_date", "report_date", "filed_at",
}


def _detect_time_col(df: pd.DataFrame) -> str | None:
    """Return the first column that looks like a timestamp, or None."""
    lower_cols = {c.lower(): c for c in df.columns}
    for alias in _TIME_COL_ALIASES:
        if alias in lower_cols:
            return lower_cols[alias]
    # fallback: any column whose dtype is datetime-like
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def time_aware_split(project_data: dict,
                     n_segments: int = 10,
                     train_segments: int = 9,
                     time_col: str | None = None
                     ) -> tuple[dict, dict]:
    """
    For each project, sort bug reports chronologically, divide into
    `n_segments` equal-size segments. First `train_segments` → train,
    remaining → test.

    If no time column is found the existing row order is used as a
    proxy for submission order (common for GitHub data exports).

    Returns
    -------
    train_data : {project: DataFrame}
    test_data  : {project: DataFrame}
    """
    train_data, test_data = {}, {}

    for project, df in project_data.items():
        df = df.copy().reset_index(drop=True)

        # Detect / sort by time column
        tcol = time_col or _detect_time_col(df)
        if tcol and tcol in df.columns:
            try:
                df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
                df = df.sort_values(tcol, na_position="first").reset_index(drop=True)
                logger.info(f"[{project}] sorted by '{tcol}'")
            except Exception as e:
                logger.warning(f"[{project}] could not parse '{tcol}': {e}. "
                               "Using row order as proxy.")
        else:
            logger.info(f"[{project}] no time column found — "
                        "using row order as chronological proxy.")

        n = len(df)
        # Build segment boundaries
        boundaries = np.linspace(0, n, n_segments + 1, dtype=int)
        train_idx = np.concatenate([
            np.arange(boundaries[i], boundaries[i + 1])
            for i in range(train_segments)
        ])
        test_idx = np.concatenate([
            np.arange(boundaries[i], boundaries[i + 1])
            for i in range(train_segments, n_segments)
        ])

        train_data[project] = df.iloc[train_idx].reset_index(drop=True)
        test_data[project]  = df.iloc[test_idx].reset_index(drop=True)

        logger.info(f"[{project}] total={n}  "
                    f"train={len(train_idx)} ({train_segments}/{n_segments})  "
                    f"test={len(test_idx)}")

    return train_data, test_data


# ======================================================================
#  Vectorise a pre-split DataFrame using an existing Word2Vec model
# ======================================================================

def vectorize_df_dict(df_dict: dict,
                      w2v_model,
                      token_cache: dict | None = None) -> dict:
    """
    Vectorize a {project: DataFrame} dict using an already-trained Word2Vec.
    `token_cache` (optional): {project: [(title_tokens, desc_tokens), ...]}
    for efficiency if tokens were already computed during W2V training.
    """
    from preprocessing import preprocess_text, vectorize_report
    wv = w2v_model.wv
    processed = {}

    for project, df in df_dict.items():
        title_vecs, desc_vecs, labels = [], [], []
        title_toks_list, desc_toks_list = [], []

        if token_cache and project in token_cache:
            pairs = token_cache[project]
        else:
            pairs = [(preprocess_text(str(r["title"])),
                      preprocess_text(str(r["description"])))
                     for _, r in df.iterrows()]

        for (tt, dt), (_, row) in zip(pairs, df.iterrows()):
            tv, dv = vectorize_report(tt, dt, wv)
            title_vecs.append(tv)
            desc_vecs.append(dv)
            labels.append(int(row["label"]))
            title_toks_list.append(tt)
            desc_toks_list.append(dt)

        processed[project] = {
            "title_vecs":   np.stack(title_vecs),
            "desc_vecs":    np.stack(desc_vecs),
            "labels":       np.array(labels),
            "title_tokens": title_toks_list,
            "desc_tokens":  desc_toks_list,
        }

    return processed


# ======================================================================
#  CNN training (single run, train → val split inside cluster)
# ======================================================================

def _metrics(labels, scores):
    binary = (np.array(scores) >= 0.5).astype(int)
    labels = np.array(labels)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    return dict(
        AUC    = auc,
        ACC    = accuracy_score(labels, binary),
        F1_v   = f1_score(labels, binary, pos_label=1, zero_division=0),
        F1_inv = f1_score(labels, binary, pos_label=0, zero_division=0),
        P_v    = precision_score(labels, binary, pos_label=1, zero_division=0),
        R_v    = recall_score(labels, binary, pos_label=1, zero_division=0),
        P_inv  = precision_score(labels, binary, pos_label=0, zero_division=0),
        R_inv  = recall_score(labels, binary, pos_label=0, zero_division=0),
    )


def train_cnn(train_data: dict, device: torch.device,
              seed: int) -> torch.nn.Module:
    """Train one CNN on pre-split train_data dict (already concatenated)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset   = BugReportDataset(train_data["title_vecs"],
                                 train_data["desc_vecs"],
                                 train_data["labels"].astype(np.float32))
    loader    = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    model     = TextCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    best_loss  = float("inf")
    patience   = 0
    best_state = None

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for titles, descs, lbls in loader:
            titles, descs, lbls = titles.to(device), descs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(titles, descs), lbls)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(lbls)
        ep_loss /= len(dataset)

        if ep_loss < best_loss:
            best_loss  = ep_loss
            patience   = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                logger.debug(f"  Early stop @ epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def infer(model: torch.nn.Module, data: dict,
          device: torch.device) -> np.ndarray:
    """Return sigmoid scores for data dict."""
    ds     = BugReportDataset(data["title_vecs"], data["desc_vecs"],
                              data["labels"].astype(np.float32))
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
    model.eval()
    scores = []
    with torch.no_grad():
        for titles, descs, _ in loader:
            scores.extend(model(titles.to(device),
                                descs.to(device)).cpu().numpy())
    return np.array(scores)


# ======================================================================
#  One full experiment run (single seed)
# ======================================================================

def concat_cluster(members: list, processed: dict) -> dict:
    """Concatenate title_vecs / desc_vecs / labels across cluster members."""
    return {
        "title_vecs": np.concatenate([processed[p]["title_vecs"] for p in members]),
        "desc_vecs":  np.concatenate([processed[p]["desc_vecs"]  for p in members]),
        "labels":     np.concatenate([processed[p]["labels"]      for p in members]),
    }


def run_single_seed(seed: int,
                    train_df: dict,
                    test_df:  dict,
                    strategy: str,
                    device:   torch.device) -> dict:
    """
    One complete CVCBR experiment for a given seed.

    Returns
    -------
    {project: {metric: value, ...}}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  SEED {seed}")
    logger.info(f"{'='*60}")

    # --- Stage 1: preprocess training data, train Word2Vec ---
    from preprocessing import train_word2vec, preprocess_text, vectorize_report
    import nltk
    # Build token cache for training data
    train_token_cache = {}
    all_sentences = []
    for project, df in train_df.items():
        pairs = [(preprocess_text(str(r["title"])),
                  preprocess_text(str(r["description"])))
                 for _, r in df.iterrows()]
        train_token_cache[project] = pairs
        for tt, dt in pairs:
            all_sentences.extend([tt, dt])

    logger.info(f"[Seed {seed}] Training Word2Vec on {len(all_sentences)} sentences ...")
    w2v_model = train_word2vec(all_sentences)
    wv = w2v_model.wv

    # Vectorize train
    processed_train = {}
    for project, df in train_df.items():
        pairs = train_token_cache[project]
        tv_list, dv_list, lb_list = [], [], []
        title_tok_list, desc_tok_list = [], []
        for (tt, dt), (_, row) in zip(pairs, df.iterrows()):
            tv, dv = vectorize_report(tt, dt, wv)
            tv_list.append(tv); dv_list.append(dv)
            lb_list.append(int(row["label"]))
            title_tok_list.append(tt); desc_tok_list.append(dt)
        processed_train[project] = {
            "title_vecs":   np.stack(tv_list),
            "desc_vecs":    np.stack(dv_list),
            "labels":       np.array(lb_list),
            "title_tokens": title_tok_list,
            "desc_tokens":  desc_tok_list,
        }

    # Vectorize test (using SAME Word2Vec — no data leakage, W2V is unsupervised)
    processed_test = vectorize_df_dict(test_df, w2v_model)

    # --- Stage 2: pre-clustering (train data only) ---
    logger.info(f"[Seed {seed}] Pre-clustering ...")
    clusters, _, _ = pre_cluster_projects(processed_train)

    # --- Stage 3: cluster adjustment (train data only) ---
    logger.info(f"[Seed {seed}] Adjustment (strategy={strategy}) ...")
    adjusted = adjust_clusters(clusters, processed_train, strategy=strategy)

    # --- Stage 4: train one CNN per cluster; evaluate on test ---
    logger.info(f"[Seed {seed}] Training CNN models ...")
    # Build a reverse map: project → cluster_id
    proj_to_cluster = {}
    for cid, members in adjusted.items():
        for p in members:
            proj_to_cluster[p] = cid

    models: dict[int, torch.nn.Module] = {}
    for cid, members in adjusted.items():
        train_concat = concat_cluster(members, processed_train)
        models[cid]  = train_cnn(train_concat, device, seed)
        nv = int(train_concat["labels"].sum())
        ni = int((train_concat["labels"] == 0).sum())
        logger.info(f"  Cluster {cid} {members}: "
                    f"train valid={nv} invalid={ni}")

    # Evaluate each project on its cluster's model
    project_results = {}
    for project, test_proc in processed_test.items():
        cid = proj_to_cluster.get(project)
        if cid is None:
            logger.warning(f"[{project}] not found in any cluster — skipping.")
            continue
        model = models[cid]
        scores = infer(model, test_proc, device)
        labels = test_proc["labels"]
        m = _metrics(labels, scores)
        m["n_test"] = len(labels)
        m["cluster"] = cid
        project_results[project] = m
        logger.info(f"  [{project}] AUC={m['AUC']:.4f}  ACC={m['ACC']:.4f}  "
                    f"F1(v)={m['F1_v']:.4f}  F1(inv)={m['F1_inv']:.4f}")

    return project_results


# ======================================================================
#  Statistical analysis
# ======================================================================

def cliffs_delta(x: list[float], y: list[float]) -> float:
    """
    Cliff's δ = (#{x > y} - #{x < y}) / (n*m)
    Measures effect size between two unpaired groups.
    Interpretation: |δ| < 0.147 negligible, < 0.33 small,
                    < 0.474 medium, >= 0.474 large.
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    gt = sum(xi > yj for xi in x for yj in y)
    lt = sum(xi < yj for xi in x for yj in y)
    return (gt - lt) / (n * m)


def bootstrap_ci(values: list[float],
                 n_boot: int = 2000,
                 alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """
    95% bootstrap confidence interval (percentile method).
    """
    rng  = np.random.RandomState(seed)
    boot = [np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_boot)]
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return round(float(lo), 4), round(float(hi), 4)


def wilcoxon_test(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test (paired).
    Returns (statistic, p_value).
    """
    if len(x) < 2 or len(x) != len(y):
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def aggregate_seeds(seed_results: list[dict]) -> dict:
    """
    Aggregate {seed: {project: {metric: value}}} across seeds.

    Returns
    -------
    {project: {metric_mean, metric_std, metric_ci_lo, metric_ci_hi, ...}}
    """
    # Collect all projects
    all_projects = set()
    for sr in seed_results:
        all_projects.update(sr.keys())

    metric_keys = ["AUC", "ACC", "F1_v", "F1_inv",
                   "P_v", "R_v", "P_inv", "R_inv"]
    aggregated = {}

    for proj in all_projects:
        vals_per_metric = defaultdict(list)
        n_test  = None
        cluster = None

        for sr in seed_results:
            if proj not in sr:
                continue
            for k in metric_keys:
                v = sr[proj].get(k, float("nan"))
                if not np.isnan(v):
                    vals_per_metric[k].append(v)
            n_test  = sr[proj].get("n_test",  n_test)
            cluster = sr[proj].get("cluster", cluster)

        proj_agg = {"n_test": n_test, "cluster": cluster}
        for k in metric_keys:
            vals = vals_per_metric[k]
            if vals:
                ci_lo, ci_hi = bootstrap_ci(vals)
                proj_agg[f"{k}_mean"]  = round(float(np.mean(vals)), 4)
                proj_agg[f"{k}_std"]   = round(float(np.std(vals)),  4)
                proj_agg[f"{k}_ci_lo"] = ci_lo
                proj_agg[f"{k}_ci_hi"] = ci_hi
            else:
                for suffix in ("_mean", "_std", "_ci_lo", "_ci_hi"):
                    proj_agg[f"{k}{suffix}"] = float("nan")

        aggregated[proj] = proj_agg

    return aggregated


def statistical_comparison(cvcbr_seed_results: list[dict],
                            baseline_seed_results: dict[str, list[dict]],
                            metric: str = "AUC") -> dict:
    """
    Compare CVCBR against each baseline using Wilcoxon + Cliff's δ.

    Parameters
    ----------
    cvcbr_seed_results   : list of {project: {metric: val}} over seeds
    baseline_seed_results: {baseline_name: list of {project: {metric: val}}}
    metric               : metric to compare (default: "AUC")

    Returns
    -------
    {baseline_name: {stat, p_value, cliffs_delta, interpretation}}
    """
    def collect_values(seed_results_list):
        """Flatten to a list of per-project-per-seed values."""
        vals = []
        for sr in seed_results_list:
            for proj, m in sr.items():
                v = m.get(metric, float("nan"))
                if not np.isnan(v):
                    vals.append(v)
        return vals

    cvcbr_vals = collect_values(cvcbr_seed_results)
    results = {}

    for name, baseline_list in baseline_seed_results.items():
        base_vals = collect_values(baseline_list)
        # Pair by position (same seed × project order)
        min_len   = min(len(cvcbr_vals), len(base_vals))
        x = cvcbr_vals[:min_len]
        y = base_vals[:min_len]

        stat, p = wilcoxon_test(x, y)
        delta   = cliffs_delta(x, y)

        # Interpret Cliff's δ
        abs_d = abs(delta)
        if abs_d < 0.147:
            interpretation = "negligible"
        elif abs_d < 0.33:
            interpretation = "small"
        elif abs_d < 0.474:
            interpretation = "medium"
        else:
            interpretation = "large"

        results[name] = {
            "wilcoxon_stat":  round(stat, 4) if not np.isnan(stat) else "nan",
            "p_value":        round(p, 6)    if not np.isnan(p)    else "nan",
            "significant":    bool(p < 0.05) if not np.isnan(p)    else False,
            "cliffs_delta":   round(delta, 4),
            "interpretation": interpretation,
        }
        sig_str = "✓" if results[name]["significant"] else "✗"
        logger.info(f"  vs {name:20s}: p={p:.4f} {sig_str}  "
                    f"δ={delta:.3f} ({interpretation})")

    return results


# ======================================================================
#  Save results
# ======================================================================

def save_all_results(aggregated: dict,
                     stat_tests: dict,
                     strategy:   str,
                     seeds:      list[int]):
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Per-project CSV
    rows = []
    metric_keys = ["AUC", "ACC", "F1_v", "F1_inv",
                   "P_v", "R_v", "P_inv", "R_inv"]
    for proj, pm in aggregated.items():
        row = {"project": proj,
               "cluster": pm.get("cluster"),
               "n_test":  pm.get("n_test")}
        for k in metric_keys:
            row[k]           = pm.get(f"{k}_mean", float("nan"))
            row[f"{k}_std"]  = pm.get(f"{k}_std",  float("nan"))
            row[f"{k}_ci_lo"]= pm.get(f"{k}_ci_lo",float("nan"))
            row[f"{k}_ci_hi"]= pm.get(f"{k}_ci_hi",float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)

    # Weighted overall
    total   = df["n_test"].fillna(0).sum()
    overall = {}
    if total > 0:
        for k in metric_keys:
            col = k
            if col in df.columns:
                overall[f"{k}_mean"] = round(
                    float((df[col].fillna(0) * df["n_test"].fillna(0)).sum() / total), 4)

    csv_path   = os.path.join(config.RESULT_DIR, f"CVCBR_{strategy}_per_project.csv")
    json_path  = os.path.join(config.RESULT_DIR, f"CVCBR_{strategy}_overall.json")
    stat_path  = os.path.join(config.RESULT_DIR, f"CVCBR_{strategy}_stats.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"seeds": seeds, "overall": overall}, f, indent=2)
    with open(stat_path, "w") as f:
        json.dump(stat_tests, f, indent=2)

    logger.info(f"Per-project CSV  → {csv_path}")
    logger.info(f"Overall JSON     → {json_path}")
    logger.info(f"Stats JSON       → {stat_path}")
    return df, overall


# ======================================================================
#  Main
# ======================================================================

def main():
    args   = parse_args()
    seeds  = [int(s.strip()) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"CVCBR  strategy={args.strategy}  seeds={seeds}  device={device}")
    logger.info(f"Time split: {args.n_segments} segments, "
                f"train={args.train_segments}, test={args.n_segments - args.train_segments}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    logger.info("\n[Step 0] Loading raw data ...")
    raw_project_data = load_project_data(args.data_dir)

    # ------------------------------------------------------------------
    # Time-aware train/test split  (done ONCE, before any seed loop)
    # ------------------------------------------------------------------
    logger.info("\n[Step 1] Time-aware train/test split ...")
    train_df, test_df = time_aware_split(
        raw_project_data,
        n_segments     = args.n_segments,
        train_segments = args.train_segments,
        time_col       = args.time_col,
    )

    # Verify test set has both classes per project
    for proj, df in test_df.items():
        vc = df["label"].value_counts()
        if len(vc) < 2:
            logger.warning(f"[{proj}] test set has only one class: {vc.to_dict()}")

    # ------------------------------------------------------------------
    # Multi-seed experiment loop
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 2] Running {len(seeds)} seeds ...")
    all_seed_results: list[dict] = []

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Seed {seed}  ({seed_idx+1}/{len(seeds)})")
        logger.info(f"{'─'*60}")
        result = run_single_seed(
            seed     = seed,
            train_df = train_df,
            test_df  = test_df,
            strategy = args.strategy,
            device   = device,
        )
        all_seed_results.append(result)

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    logger.info("\n[Step 3] Aggregating results across seeds ...")
    aggregated = aggregate_seeds(all_seed_results)

    # ------------------------------------------------------------------
    # Statistical tests (placeholder — populate baseline_results
    # with actual baseline runs to enable comparison)
    # ------------------------------------------------------------------
    logger.info("\n[Step 4] Statistical analysis ...")
    # NOTE: To compare against baselines (CNN / BERT / Fine-tuned BERT),
    # run those baselines and pass their seed results here.
    # For now we compute self-statistics as a demonstration.
    baseline_results: dict[str, list[dict]] = {}
    # Example: baseline_results["CNN"] = cnn_seed_results
    stat_tests = {}
    if baseline_results:
        logger.info("Wilcoxon signed-rank test (CVCBR vs baselines) on AUC:")
        stat_tests = statistical_comparison(
            all_seed_results, baseline_results, metric="AUC"
        )
    else:
        logger.info("  No baseline results provided — skipping statistical tests.")
        logger.info("  (Add baseline seed results to 'baseline_results' dict in main.py)")

    # Bootstrap CI on overall AUC across seeds
    all_aucs = []
    for sr in all_seed_results:
        for proj, m in sr.items():
            v = m.get("AUC", float("nan"))
            if not np.isnan(v):
                all_aucs.append(v)
    if all_aucs:
        ci_lo, ci_hi = bootstrap_ci(all_aucs)
        logger.info(f"\n  Overall AUC 95% CI: [{ci_lo}, {ci_hi}]")
        stat_tests["overall_AUC_CI"] = {"ci_lo": ci_lo, "ci_hi": ci_hi,
                                         "mean": round(float(np.mean(all_aucs)), 4)}

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info("\n[Step 5] Saving results ...")
    df, overall = save_all_results(aggregated, stat_tests, args.strategy, seeds)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info(f"Final Results  (strategy={args.strategy}, "
                f"{len(seeds)} seeds × time-aware split)")
    logger.info("-" * 60)
    for k, v in overall.items():
        logger.info(f"  {k:15s}: {v}")
    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    return aggregated, overall


if __name__ == "__main__":
    main()

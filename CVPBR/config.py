# ============================================================
# CVCBR Configuration
# Paper: "CVCBR: Clustering-based Validity Classification of Bug Reports"
# ============================================================

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# --- Data Preprocessing ---
TITLE_MAX_LEN = 10          # retain first 10 words from title
DESC_MAX_LEN = 50           # retain first 50 words from description
WORD2VEC_DIM = 200          # embedding dimension
WORD2VEC_WINDOW = 5         # context window size
WORD2VEC_MIN_COUNT = 5      # min word frequency
WORD2VEC_WORKERS = 4
WORD2VEC_SG = 1             # 1 = Skip-gram

# --- Spectral Clustering ---
CLUSTER_K_RANGE = None      # None => auto-detect from [1, n_projects]
# Weighted score weights (Section 3.2, Formula 1)
W_SILHOUETTE = 0.7
W_CH = 0.15
W_DB = 0.15

# --- Cluster Adjustment (Algorithm 2 & 3) ---
MIN_CLUSTER_SIZE = 3000     # N: minimum intra-cluster sample size (Section 4.5)
R_MAX = 1.6                 # positive-to-negative ratio upper bound (Section 4.5)
R_MIN = 1 / R_MAX           # => [1/1.6, 1.6]
INTERFERENCE_ALPHA = 0.05   # threshold α for interference ratio (Section 3.3.2)
TOP_WORD_PERCENT = 0.05     # top 5% word frequency for interference ratio

# --- CNN Model (Section 3.4) ---
TITLE_KERNEL_SIZES = [1, 2, 3]
DESC_KERNEL_SIZES = [2, 3, 4]
NUM_FILTERS = 128
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5  # stop if loss does not drop for 5 epochs

# --- Evaluation / Experiment Protocol ---
# Time-aware split (Section 4)
N_SEGMENTS     = 10    # divide each project into 10 equal time segments
TRAIN_SEGMENTS = 9     # first 9 → train, last 1 → test

# Multi-seed (Section 4)
RANDOM_SEEDS = [42, 0, 1, 2, 3]   # 5 seeds
RANDOM_SEED  = 42                  # default single seed

# Bootstrap CI
N_BOOTSTRAP = 2000
CI_ALPHA    = 0.05   # 95% CI

import os
import re
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK data (only first time)
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Column name aliases: maps common variants → canonical name
_COL_ALIASES = {
    "title":       {"title", "bug_title", "summary", "bug_summary", "subject"},
    "description": {"description", "desc", "body", "content", "bug_description",
                    "bug_body", "details", "text"},
    "label":       {"label", "validity", "valid", "is_valid", "status",
                    "class", "target", "y"},
}



def _read_file(fpath: str) -> pd.DataFrame:
    """Read a single data file (xlsx or csv) into a DataFrame."""
    ext = os.path.splitext(fpath)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(fpath, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(fpath)
    else:
        raise ValueError(f"Unsupported file format: {fpath}")
    return df


def _resolve_columns(df: pd.DataFrame, project: str) -> pd.DataFrame | None:

    col_map = {c.strip().lower(): c for c in df.columns}   # lower → original

    resolved = {}
    for canonical, aliases in _COL_ALIASES.items():
        match = aliases & set(col_map.keys())
        if match:
            resolved[canonical] = col_map[next(iter(match))]

    missing = {"title", "description", "label"} - set(resolved.keys())
    if missing:
        logger.warning(f"[{project}] cannot resolve columns {missing}. "
                       f"Available: {list(col_map.keys())}. Skipping.")
        return None

    df = df[[resolved["title"], resolved["description"], resolved["label"]]].copy()
    df.columns = ["title", "description", "label"]
    return df


def _encode_label(series: pd.Series, project: str) -> pd.Series:
    """
    Convert label column to int {0, 1}.
    Handles: int, float, bool, or string ('valid'/'invalid', '1'/'0', 'true'/'false').
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    # String mapping
    mapping = {
        "valid": 1, "invalid": 0,
        "true": 1,  "false": 0,
        "yes": 1,   "no": 0,
        "1": 1,     "0": 0,
    }
    lower = series.astype(str).str.strip().str.lower()
    encoded = lower.map(mapping)
    if encoded.isna().any():
        unique_vals = lower.unique().tolist()
        logger.warning(f"[{project}] unrecognised label values: {unique_vals}. "
                       "Dropping those rows.")
        encoded = encoded.dropna()
    return encoded.astype(int)


def load_project_data(data_dir: str) -> dict[str, pd.DataFrame]:
 
    project_data = {}
    supported = (".xlsx", ".xls", ".xlsm", ".csv")
    files = [f for f in os.listdir(data_dir)
             if os.path.splitext(f)[1].lower() in supported]

    if not files:
        raise FileNotFoundError(
            f"No XLSX/CSV files found in '{data_dir}'. "
            "Place one file per project there."
        )

    for fname in sorted(files):
        project = os.path.splitext(fname)[0]
        fpath = os.path.join(data_dir, fname)
        try:
            df = _read_file(fpath)
        except Exception as e:
            logger.warning(f"[{project}] failed to read '{fname}': {e}. Skipping.")
            continue

        df = _resolve_columns(df, project)
        if df is None:
            continue

        # Drop rows with NaN in any required field
        before = len(df)
        df = df.dropna(subset=["title", "description", "label"])
        if len(df) < before:
            logger.warning(f"[{project}] dropped {before - len(df)} rows with NaN.")

        df["label"] = _encode_label(df["label"], project)
        df["project"] = project
        df = df.reset_index(drop=True)

        n_valid   = int(df["label"].sum())
        n_invalid = int((df["label"] == 0).sum())
        logger.info(f"Loaded [{project:30s}]: {len(df):5d} reports  "
                    f"valid={n_valid}  invalid={n_invalid}")
        project_data[project] = df

    if not project_data:
        raise RuntimeError("No valid project data loaded. Check your files and column names.")

    return project_data



def normalize(text: str) -> str:
    """Lowercase; keep only alphabetic characters and spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


def tokenize_and_remove_stopwords(text: str) -> list[str]:
    """Tokenize with NLTK word_tokenize, remove stop words."""
    tokens = word_tokenize(text)
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]


def preprocess_text(text: str) -> list[str]:
    """Full pipeline: normalize → tokenize → remove stopwords."""
    return tokenize_and_remove_stopwords(normalize(text))


def train_word2vec(sentences: list[list[str]], save_path: str = None) -> Word2Vec:
    logger.info("Training Word2Vec model ...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=config.WORD2VEC_DIM,
        window=config.WORD2VEC_WINDOW,
        min_count=config.WORD2VEC_MIN_COUNT,
        workers=config.WORD2VEC_WORKERS,
        sg=config.WORD2VEC_SG,
    )
    if save_path:
        model.save(save_path)
        logger.info(f"Word2Vec saved to {save_path}")
    return model



def tokens_to_matrix(tokens: list[str], max_len: int, wv) -> np.ndarray:

    dim = wv.vector_size
    matrix = np.zeros((max_len, dim), dtype=np.float32)
    for i, token in enumerate(tokens[:max_len]):
        if token in wv:
            matrix[i] = wv[token]
    return matrix


def vectorize_report(title_tokens: list[str],
                     desc_tokens: list[str],
                     wv) -> tuple[np.ndarray, np.ndarray]:
    """Return (title_vec, desc_vec) as numpy arrays."""
    title_vec = tokens_to_matrix(title_tokens, config.TITLE_MAX_LEN, wv)
    desc_vec = tokens_to_matrix(desc_tokens, config.DESC_MAX_LEN, wv)
    return title_vec, desc_vec



def preprocess_all(project_data: dict[str, pd.DataFrame],
                   w2v_save_path: str = None
                   ) -> tuple[dict, Word2Vec]:

    # --- collect all sentences for W2V training ---
    all_sentences = []
    token_cache = {}   # project -> list of (title_tokens, desc_tokens)

    for project, df in project_data.items():
        pairs = []
        for _, row in df.iterrows():
            tt = preprocess_text(str(row["title"]))
            dt = preprocess_text(str(row["description"]))
            pairs.append((tt, dt))
            all_sentences.extend([tt, dt])
        token_cache[project] = pairs

    # --- train Word2Vec on all projects jointly ---
    w2v_model = train_word2vec(all_sentences, save_path=w2v_save_path)
    wv = w2v_model.wv

    # --- vectorize ---
    processed = {}
    for project, df in project_data.items():
        pairs = token_cache[project]
        title_vecs, desc_vecs, labels = [], [], []
        for (tt, dt), (_, row) in zip(pairs, df.iterrows()):
            tv, dv = vectorize_report(tt, dt, wv)
            title_vecs.append(tv)
            desc_vecs.append(dv)
            labels.append(int(row["label"]))

        processed[project] = {
            "title_vecs": np.stack(title_vecs),    # (N, 10, 200)
            "desc_vecs":  np.stack(desc_vecs),     # (N, 50, 200)
            "labels":     np.array(labels),
            "title_tokens": [p[0] for p in pairs],
            "desc_tokens":  [p[1] for p in pairs],
        }
        logger.info(f"Vectorized [{project}]: shape title={processed[project]['title_vecs'].shape}")

    return processed, w2v_model


def subset_data(data: dict, idx: np.ndarray) -> dict:
    """Return a sub-dict with only the given indices."""
    return {
        k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
        for k, v in data.items()
    }


if __name__ == "__main__":
    project_data = load_project_data(config.DATA_DIR)
    processed, w2v = preprocess_all(
        project_data,
        w2v_save_path=os.path.join(config.MODEL_DIR, "word2vec.model")
    )
    logger.info("Preprocessing complete.")
    for p, d in processed.items():
        logger.info(f"  {p}: {d['title_vecs'].shape[0]} samples")

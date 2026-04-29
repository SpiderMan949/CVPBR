import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)



class BugReportDataset(Dataset):
    def __init__(self, title_vecs: np.ndarray, desc_vecs: np.ndarray, labels: np.ndarray):

        self.titles = torch.tensor(title_vecs, dtype=torch.float32)
        self.descs  = torch.tensor(desc_vecs,  dtype=torch.float32)
        self.labels = torch.tensor(labels,     dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.titles[idx], self.descs[idx], self.labels[idx]



class TextCNN(nn.Module):


    def __init__(self,
                 embed_dim: int = config.WORD2VEC_DIM,
                 title_len: int = config.TITLE_MAX_LEN,
                 desc_len:  int = config.DESC_MAX_LEN,
                 title_kernels: list = config.TITLE_KERNEL_SIZES,
                 desc_kernels:  list = config.DESC_KERNEL_SIZES,
                 n_filters: int = config.NUM_FILTERS,
                 dropout: float = config.DROPOUT_RATE):
        super().__init__()

        # Title convolutions
        # Input shape: (batch, title_len, embed_dim) → permute to (batch, embed_dim, title_len)
        self.title_convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=n_filters,
                      kernel_size=k,
                      padding=0)
            for k in title_kernels
        ])

        # Desc convolutions
        self.desc_convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=n_filters,
                      kernel_size=k,
                      padding=0)
            for k in desc_kernels
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        total_filters = n_filters * (len(title_kernels) + len(desc_kernels))
        self.fc = nn.Linear(total_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def _conv_pool(self, x: torch.Tensor, convs: nn.ModuleList) -> torch.Tensor:
        """Apply each conv → ReLU → GlobalMaxPool; concatenate results."""
        # x: (batch, seq_len, embed_dim) → (batch, embed_dim, seq_len)
        x = x.permute(0, 2, 1)
        pooled = []
        for conv in convs:
            c = self.relu(conv(x))          # (batch, n_filters, L')
            p = c.max(dim=-1).values        # (batch, n_filters)
            pooled.append(p)
        return torch.cat(pooled, dim=1)     # (batch, n_filters * n_kernels)

    def forward(self, title: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        title_feat = self._conv_pool(title, self.title_convs)
        desc_feat  = self._conv_pool(desc,  self.desc_convs)
        fused = self.dropout(torch.cat([title_feat, desc_feat], dim=1))
        out = self.sigmoid(self.fc(fused))  # (batch, 1)
        return out.squeeze(1)



def train_one_cluster(cluster_id: int,
                      members: list[str],
                      processed_data: dict,
                      device: torch.device,
                      save_dir: str = config.MODEL_DIR
                      ) -> tuple[nn.Module, dict]:
    """
    Train one CNN model for the given cluster.
    Returns (model, val_metrics).
    """
    # Aggregate data from all projects in the cluster
    all_titles, all_descs, all_labels = [], [], []
    for p in members:
        all_titles.append(processed_data[p]["title_vecs"])
        all_descs.append(processed_data[p]["desc_vecs"])
        all_labels.append(processed_data[p]["labels"])

    title_vecs = np.concatenate(all_titles, axis=0)
    desc_vecs  = np.concatenate(all_descs,  axis=0)
    labels     = np.concatenate(all_labels, axis=0)

    # Train / val split (80/20)
    idx = np.arange(len(labels))
    tr_idx, va_idx = train_test_split(idx, test_size=config.TEST_SPLIT,
                                      random_state=config.RANDOM_SEED,
                                      stratify=labels)

    tr_set = BugReportDataset(title_vecs[tr_idx], desc_vecs[tr_idx], labels[tr_idx])
    va_set = BugReportDataset(title_vecs[va_idx], desc_vecs[va_idx], labels[va_idx])

    tr_loader = DataLoader(tr_set, batch_size=config.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=config.BATCH_SIZE, shuffle=False)

    model = TextCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info(f"Training Cluster {cluster_id} | projects={members} | "
                f"train={len(tr_set)} val={len(va_set)}")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        # --- train ---
        model.train()
        tr_loss = 0.0
        for titles, descs, lbls in tr_loader:
            titles, descs, lbls = titles.to(device), descs.to(device), lbls.to(device)
            optimizer.zero_grad()
            preds = model(titles, descs)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(lbls)
        tr_loss /= len(tr_set)

        # --- validate ---
        model.eval()
        va_loss = 0.0
        all_preds, all_lbls = [], []
        with torch.no_grad():
            for titles, descs, lbls in va_loader:
                titles, descs, lbls = titles.to(device), descs.to(device), lbls.to(device)
                preds = model(titles, descs)
                va_loss += criterion(preds, lbls).item() * len(lbls)
                all_preds.extend(preds.cpu().numpy())
                all_lbls.extend(lbls.cpu().numpy())
        va_loss /= len(va_set)

        logger.info(f"  Epoch {epoch:3d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        # --- early stopping ---
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"cluster_{cluster_id}.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved model → {ckpt_path}")

    # Final val metrics
    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for titles, descs, lbls in va_loader:
            titles, descs = titles.to(device), descs.to(device)
            preds = model(titles, descs)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.numpy())

    binary_preds = (np.array(all_preds) >= 0.5).astype(int)
    metrics = {
        "auc":    round(roc_auc_score(all_lbls, all_preds), 4),
        "acc":    round(accuracy_score(all_lbls, binary_preds), 4),
        "f1_v":   round(f1_score(all_lbls, binary_preds, pos_label=1, zero_division=0), 4),
        "f1_inv": round(f1_score(all_lbls, binary_preds, pos_label=0, zero_division=0), 4),
        "p_v":    round(precision_score(all_lbls, binary_preds, pos_label=1, zero_division=0), 4),
        "r_v":    round(recall_score(all_lbls, binary_preds, pos_label=1, zero_division=0), 4),
        "p_inv":  round(precision_score(all_lbls, binary_preds, pos_label=0, zero_division=0), 4),
        "r_inv":  round(recall_score(all_lbls, binary_preds, pos_label=0, zero_division=0), 4),
    }
    logger.info(f"Cluster {cluster_id} val metrics: {metrics}")
    return model, metrics


def train_all_clusters(adjusted_clusters: dict[int, list[str]],
                       processed_data: dict) -> tuple[dict[int, nn.Module], dict]:
    """
    Train one CNN per cluster.
    Returns (models, all_metrics).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    models = {}
    all_metrics = {}

    for cid, members in adjusted_clusters.items():
        model, metrics = train_one_cluster(cid, members, processed_data, device)
        models[cid] = model
        all_metrics[cid] = {"projects": members, **metrics}

    return models, all_metrics


if __name__ == "__main__":
    from preprocessing import load_project_data, preprocess_all
    from clustering import pre_cluster_projects
    from adjustment import adjust_clusters
    logging.basicConfig(level=logging.INFO)

    project_data = load_project_data(config.DATA_DIR)
    processed, _ = preprocess_all(project_data)
    clusters, _, _ = pre_cluster_projects(processed)
    adjusted = adjust_clusters(clusters, processed, strategy="S3")
    models, metrics = train_all_clusters(adjusted, processed)
    for cid, m in metrics.items():
        print(f"Cluster {cid}: {m}")

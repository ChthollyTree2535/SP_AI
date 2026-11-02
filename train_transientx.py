import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_CROP = (68, 184, 647, 950)  # left, top, right, bottom for 681x1021 TransientX PNGs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test a high-recall TransientX single-pulse image classifier."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing class folders 0 and 1.")
    parser.add_argument("--output-dir", default="runs", help="Directory for run artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--model-arch",
        choices=("diagnostic_resnet_v2", "small_resnet"),
        default="diagnostic_resnet_v2",
        help="Classifier backbone. diagnostic_resnet_v2 is the high-resolution default.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=24,
        help="Base width for diagnostic_resnet_v2. Ignored by small_resnet.",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--crop", type=int, nargs=4, default=DEFAULT_CROP, metavar=("L", "T", "R", "B"))
    parser.add_argument("--crop-jitter", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-pos-fraction", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, mps, etc.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device_arg="auto", cpu=False):
    if cpu or device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device_arg == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
    return torch.device(device_arg)


def scan_dataset(data_dir):
    data_dir = Path(data_dir)
    items = []
    for label_name in ("0", "1"):
        class_dir = data_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        label = int(label_name)
        for path in sorted(class_dir.glob("*.png")):
            items.append({"path": str(path.resolve()), "label": label})
    if not items:
        raise RuntimeError(f"No PNG files found under {data_dir}")
    return items


def stratified_split(items, val_frac, test_frac, seed):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for item in items:
        by_label[item["label"]].append(item)

    split = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)
        n_total = len(group)
        n_test = max(1, int(round(n_total * test_frac)))
        n_val = max(1, int(round(n_total * val_frac)))
        if n_test + n_val >= n_total:
            raise ValueError(f"Not enough samples for label {label} to split train/val/test.")
        split["test"].extend(group[:n_test])
        split["val"].extend(group[n_test : n_test + n_val])
        split["train"].extend(group[n_test + n_val :])

    for values in split.values():
        rng.shuffle(values)
    return split


class TransientXDataset(Dataset):
    def __init__(self, items, crop, size, train=False, crop_jitter=0):
        self.items = items
        self.crop = tuple(crop)
        self.size = tuple(size)
        self.train = train
        self.crop_jitter = int(crop_jitter)

    def __len__(self):
        return len(self.items)

    def _jitter_crop(self, image_shape):
        left, top, right, bottom = self.crop
        if self.train and self.crop_jitter > 0:
            h, w = image_shape[:2]
            jitter = self.crop_jitter
            dl = random.randint(-jitter, jitter)
            dt = random.randint(-jitter, jitter)
            dr = random.randint(-jitter, jitter)
            db = random.randint(-jitter, jitter)
            left = min(max(0, left + dl), w - 2)
            top = min(max(0, top + dt), h - 2)
            right = min(max(left + 1, right + dr), w)
            bottom = min(max(top + 1, bottom + db), h)
        return left, top, right, bottom

    def __getitem__(self, idx):
        item = self.items[idx]
        image = imageio.imread(item["path"])
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        left, top, right, bottom = self._jitter_crop(image.shape)
        image = image[top:bottom, left:right, :].astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
        ).squeeze(0)

        if self.train:
            contrast = 1.0 + random.uniform(-0.12, 0.12)
            brightness = random.uniform(-0.06, 0.06)
            tensor = (tensor - 0.5) * contrast + 0.5 + brightness
            if random.random() < 0.35:
                tensor = tensor + torch.randn_like(tensor) * random.uniform(0.005, 0.025)
            tensor = tensor.clamp(0.0, 1.0)

        tensor = (tensor - 0.5) / 0.5
        label = torch.tensor(float(item["label"]), dtype=torch.float32)
        return tensor, label, item["path"]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.silu(out + identity)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        weights = self.pool(x)
        weights = F.silu(self.fc1(weights))
        weights = torch.sigmoid(self.fc2(weights))
        return x * weights


class DiagnosticResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.silu(out + identity)


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 32))
        self.stage2 = nn.Sequential(ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64))
        self.stage3 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128, dropout=0.05))
        self.stage4 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256, dropout=0.08))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x).squeeze(1)


class DiagnosticResNetV2(nn.Module):
    """Residual classifier tuned for the multi-panel TransientX diagnostic ROI."""

    def __init__(self, base_channels=24):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DiagnosticResidualBlock(c1, c1),
            DiagnosticResidualBlock(c1, c1, dropout=0.02),
        )
        self.stage2 = nn.Sequential(
            DiagnosticResidualBlock(c1, c2, stride=2),
            DiagnosticResidualBlock(c2, c2, dropout=0.03),
        )
        self.stage3 = nn.Sequential(
            DiagnosticResidualBlock(c2, c3, stride=2),
            DiagnosticResidualBlock(c3, c3, dropout=0.05),
        )
        self.stage4 = nn.Sequential(
            DiagnosticResidualBlock(c3, c4, stride=2),
            DiagnosticResidualBlock(c4, c4, dilation=2, dropout=0.08),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(c4, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x).squeeze(1)


def build_model(model_arch="diagnostic_resnet_v2", base_channels=24):
    if model_arch == "small_resnet":
        return SmallResNet()
    if model_arch == "diagnostic_resnet_v2":
        return DiagnosticResNetV2(base_channels=base_channels)
    raise ValueError(f"Unknown model architecture: {model_arch}")


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


def make_train_sampler(items, target_pos_fraction):
    labels = np.array([item["label"] for item in items], dtype=np.int64)
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    weights = np.zeros(len(items), dtype=np.float64)
    weights[labels == 1] = target_pos_fraction / n_pos
    weights[labels == 0] = (1.0 - target_pos_fraction) / n_neg
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(items),
        replacement=True,
    )


def average_precision(labels, scores):
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    total_pos = labels_sorted.sum()
    if total_pos == 0:
        return 0.0
    cumsum = np.cumsum(labels_sorted)
    precision_at_k = cumsum / (np.arange(len(labels_sorted)) + 1)
    return float((precision_at_k * labels_sorted).sum() / total_pos)


def compute_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    metrics = {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "average_precision": average_precision(labels, scores),
    }
    order = np.argsort(-scores)
    for pct in (1, 2, 5, 10):
        k = max(1, int(math.ceil(len(labels) * pct / 100.0)))
        hits = int(labels[order[:k]].sum())
        metrics[f"recall_at_top_{pct}pct"] = hits / max(1, int(labels.sum()))
        metrics[f"positives_at_top_{pct}pct"] = hits
        metrics[f"k_top_{pct}pct"] = k
    return metrics


def save_manifest(path, split):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        for split_name, items in split.items():
            for item in items:
                writer.writerow({"path": item["path"], "label": item["label"], "split": split_name})


def save_predictions(path, labels, scores, paths):
    order = np.argsort(-np.asarray(scores))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "path", "label", "prob_real", "percentile"])
        writer.writeheader()
        n = len(paths)
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "path": paths[idx],
                    "label": int(labels[idx]),
                    "prob_real": f"{scores[idx]:.8f}",
                    "percentile": f"{rank / max(1, n):.8f}",
                }
            )


def save_threshold_table(path, labels, scores):
    rows = []
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos_scores = sorted(scores[labels == 1], reverse=True)
    for target_recall in (0.90, 0.95, 0.98, 0.99, 1.00):
        if not pos_scores:
            threshold = 1.0
        else:
            needed = max(1, int(math.ceil(len(pos_scores) * target_recall)))
            threshold = float(pos_scores[min(needed - 1, len(pos_scores) - 1)])
        preds = scores >= threshold
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        selected = int(preds.sum())
        rows.append(
            {
                "target_recall": target_recall,
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "selected": selected,
                "selected_fraction": selected / max(1, len(labels)),
                "precision": tp / max(1, tp + fp),
                "actual_recall": tp / max(1, int(labels.sum())),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_plots(run_dir, log_rows, test_labels, test_scores, test_metrics, threshold_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in log_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in log_rows], marker="o", label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in log_rows], marker="o", label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row["train_ap"] for row in log_rows], marker="o", label="train AP")
    axes[0, 1].plot(epochs, [row["val_ap"] for row in log_rows], marker="o", label="val AP")
    axes[0, 1].set_title("Average Precision")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["val_recall"] for row in log_rows], marker="o", label="threshold recall")
    axes[1, 0].plot(epochs, [row["val_recall_top5"] for row in log_rows], marker="o", label="recall@top5%")
    axes[1, 0].plot(epochs, [row.get("val_recall_top10", 0.0) for row in log_rows], marker="o", label="recall@top10%")
    axes[1, 0].set_title("Validation Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [row["selection_score"] for row in log_rows], marker="o")
    axes[1, 1].set_title("Model Selection Score")
    axes[1, 1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = [1, 2, 5, 10]
    recalls = [test_metrics[f"recall_at_top_{k}pct"] for k in ks]
    ax.bar([f"Top {k}%" for k in ks], recalls, color="#3b82f6")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recall")
    ax.set_title("Test Recall By Review Budget")
    for i, value in enumerate(recalls):
        ax.text(i, min(1.0, value + 0.02), f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(run_dir / "topk_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    matrix = np.array([[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]])
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title("Test Confusion Matrix @ 0.5")
    for y in range(2):
        for x in range(2):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    labels = np.asarray(test_labels, dtype=np.int64)
    scores = np.asarray(test_scores, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores[labels == 0], bins=40, alpha=0.65, label="negative", color="#64748b")
    ax.hist(scores[labels == 1], bins=40, alpha=0.65, label="positive", color="#ef4444")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold 0.5")
    ax.set_xlabel("Predicted probability of real pulse")
    ax.set_ylabel("Count")
    ax.set_title("Test Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "score_histogram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    targets = [row["target_recall"] for row in threshold_rows]
    selected = [row["selected_fraction"] for row in threshold_rows]
    precision = [row["precision"] for row in threshold_rows]
    ax.plot(targets, selected, marker="o", label="selected fraction")
    ax.plot(targets, precision, marker="o", label="precision")
    ax.set_xlabel("Target recall")
    ax.set_ylim(0, 1.02)
    ax.set_title("Threshold Tradeoff On Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "threshold_tradeoff.png", dpi=160)
    plt.close(fig)


def image_to_network_input(path, crop, size):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    left, top, right, bottom = crop
    roi = image[top:bottom, left:right, :].astype(np.float32) / 255.0
    tensor = torch.from_numpy(roi).permute(2, 0, 1).contiguous()
    resized = F.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)
    return roi, resized.permute(1, 2, 0).numpy()


def save_input_preview(run_dir, split, crop, size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = []
    for label, label_name in ((1, "positive"), (0, "negative")):
        sample = next(
            (item for item in split["train"] if item["label"] == label),
            next((item for item in split["val"] if item["label"] == label), None),
        )
        if sample is not None:
            samples.append((label_name, sample))
    if not samples:
        return

    fig, axes = plt.subplots(len(samples), 2, figsize=(8.2, 4.4 * len(samples)))
    axes = np.atleast_2d(axes)
    for row, (label_name, sample) in enumerate(samples):
        roi, resized = image_to_network_input(sample["path"], crop, size)
        axes[row, 0].imshow(roi)
        axes[row, 0].set_title(f"{label_name}: ROI crop {roi.shape[1]} x {roi.shape[0]}")
        axes[row, 1].imshow(resized)
        axes[row, 1].set_title(f"{label_name}: network input {size[1]} x {size[0]}")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle("TransientX ROI Coverage And High-Resolution Resize", y=0.995)
    fig.tight_layout()
    fig.savefig(run_dir / "input_preview.png", dpi=180)
    plt.close(fig)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_scores = []
    all_paths = []
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_count += images.size(0)
        all_labels.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
        all_scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_paths.extend(list(paths))
    return total_loss / max(1, total_count), all_labels, all_scores, all_paths


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, cpu=args.cpu)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"transientx_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    items = scan_dataset(args.data_dir)
    split = stratified_split(items, args.val_frac, args.test_frac, args.seed)
    save_manifest(run_dir / "manifest.csv", split)

    size = (args.height, args.width)
    save_input_preview(run_dir, split, args.crop, size)
    datasets = {
        name: TransientXDataset(
            values,
            crop=args.crop,
            size=size,
            train=(name == "train"),
            crop_jitter=args.crop_jitter if name == "train" else 0,
        )
        for name, values in split.items()
    }
    sampler = make_train_sampler(split["train"], args.train_pos_fraction)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model_arch, args.base_channels).to(device)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    metadata = {
        "args": vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "model_arch": args.model_arch,
        "trainable_parameters": count_trainable_parameters(model),
        "class_counts": {
            name: {
                "total": len(values),
                "positive": int(sum(item["label"] for item in values)),
                "negative": int(sum(1 - item["label"] for item in values)),
            }
            for name, values in split.items()
        },
        "run_dir": str(run_dir.resolve()),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    log_rows = []

    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_labels, train_scores, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_labels, val_scores, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        train_metrics = compute_metrics(train_labels, train_scores, args.threshold)
        val_metrics = compute_metrics(val_labels, val_scores, args.threshold)
        selection_score = (
            0.45 * val_metrics["average_precision"]
            + 0.20 * val_metrics["recall_at_top_5pct"]
            + 0.35 * val_metrics["recall_at_top_10pct"]
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ap": train_metrics["average_precision"],
            "val_ap": val_metrics["average_precision"],
            "val_recall": val_metrics["recall"],
            "val_recall_top1": val_metrics["recall_at_top_1pct"],
            "val_recall_top2": val_metrics["recall_at_top_2pct"],
            "val_recall_top5": val_metrics["recall_at_top_5pct"],
            "val_recall_top10": val_metrics["recall_at_top_10pct"],
            "selection_score": selection_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)
        print(
            "epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            "val_ap={val_ap:.4f} val_r@1%={val_recall_top1:.4f} "
            "val_r@2%={val_recall_top2:.4f} val_r@5%={val_recall_top5:.4f} "
            "val_r@10%={val_recall_top10:.4f}".format(**row),
            flush=True,
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": metadata,
                    "epoch": epoch,
                    "selection_score": selection_score,
                },
                run_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
                break

        with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_labels, test_scores, test_paths = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    test_metrics = compute_metrics(test_labels, test_scores, args.threshold)
    test_metrics["test_loss"] = test_loss
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_selection_score"] = best_score

    save_predictions(run_dir / "test_predictions.csv", test_labels, test_scores, test_paths)
    threshold_rows = save_threshold_table(run_dir / "threshold_table.csv", test_labels, test_scores)
    save_plots(run_dir, log_rows, test_labels, test_scores, test_metrics, threshold_rows)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("TEST " + json.dumps(test_metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"RUN_DIR {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

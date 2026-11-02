import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_transientx import (
    FocalLoss,
    TransientXDataset,
    build_model,
    compute_metrics,
    run_epoch,
    save_plots,
    save_predictions,
    save_threshold_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate and regenerate artifacts for a run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_manifest(run_dir):
    split = {"train": [], "val": [], "test": []}
    with open(run_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split[row["split"]].append({"path": row["path"], "label": int(row["label"])})
    return split


def load_log(run_dir):
    with open(run_dir / "training_log.csv", newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                parsed[key] = int(value) if key == "epoch" else float(value)
            rows.append(parsed)
    return rows


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    train_args = config["args"]
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    split = load_manifest(run_dir)

    dataset = TransientXDataset(
        split["test"],
        crop=train_args["crop"],
        size=(train_args["height"], train_args["width"]),
        train=False,
        crop_jitter=0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        train_args.get("model_arch", "small_resnet"),
        train_args.get("base_channels", 24),
    ).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    criterion = FocalLoss(alpha=train_args["focal_alpha"], gamma=train_args["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args["lr"])
    test_loss, labels, scores, paths = run_epoch(
        model, loader, criterion, optimizer, device, train=False
    )
    metrics = compute_metrics(labels, scores, train_args["threshold"])
    metrics["test_loss"] = test_loss
    metrics["best_epoch"] = checkpoint.get("epoch")
    metrics["best_selection_score"] = checkpoint.get("selection_score")

    save_predictions(run_dir / "test_predictions.csv", labels, scores, paths)
    threshold_rows = save_threshold_table(run_dir / "threshold_table.csv", labels, scores)
    save_plots(run_dir, load_log(run_dir), labels, scores, metrics, threshold_rows)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"RUN_DIR {run_dir.resolve()}")


if __name__ == "__main__":
    main()

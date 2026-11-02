import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize several TransientX training runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to compare.")
    parser.add_argument("--output-dir", default="runs", help="Directory for summary CSV and PNG.")
    parser.add_argument("--name", default="tuning_summary", help="Output file prefix.")
    return parser.parse_args()


def load_last_log(run_dir):
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        return {}
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for run in args.runs:
        run_dir = Path(run)
        with open(run_dir / "metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(run_dir / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        train_args = config["args"]
        last = load_last_log(run_dir)
        summary_rows.append(
            {
                "run": run_dir.name,
                "model_arch": train_args.get("model_arch", "small_resnet"),
                "base_channels": train_args.get("base_channels", ""),
                "height": train_args.get("height"),
                "width": train_args.get("width"),
                "epochs": train_args.get("epochs"),
                "train_pos_fraction": train_args.get("train_pos_fraction"),
                "focal_alpha": train_args.get("focal_alpha"),
                "focal_gamma": train_args.get("focal_gamma"),
                "best_epoch": metrics.get("best_epoch"),
                "test_ap": metrics.get("average_precision"),
                "test_precision": metrics.get("precision"),
                "test_recall": metrics.get("recall"),
                "test_f1": metrics.get("f1"),
                "test_recall_top5": metrics.get("recall_at_top_5pct"),
                "test_recall_top10": metrics.get("recall_at_top_10pct"),
                "val_ap_last": last.get("val_ap", ""),
                "val_recall_top10_last": last.get("val_recall_top10", ""),
            }
        )

    csv_path = output_dir / f"{args.name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["run"] for row in summary_rows]
    x = range(len(summary_rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.22
    ax.bar([i - width for i in x], [float(row["test_ap"]) for row in summary_rows], width, label="AP")
    ax.bar([i for i in x], [float(row["test_recall"]) for row in summary_rows], width, label="Recall@0.5")
    ax.bar(
        [i + width for i in x],
        [float(row["test_recall_top10"]) for row in summary_rows],
        width,
        label="Recall@Top10%",
    )
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title("Tuning Run Comparison")
    ax.legend()
    fig.tight_layout()
    png_path = output_dir / f"{args.name}.png"
    fig.savefig(png_path, dpi=160)
    print(f"summary_csv={csv_path.resolve()}")
    print(f"summary_png={png_path.resolve()}")


if __name__ == "__main__":
    main()

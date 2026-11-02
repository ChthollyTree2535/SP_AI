#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy strictly thresholded TransientX PNGs into a flat directory without renaming them."
    )
    parser.add_argument("--scores", required=True, help="CSV written by predict_transientx_streaming.py.")
    parser.add_argument("--output-dir", required=True, help="Flat destination directory for selected PNGs.")
    parser.add_argument("--min-prob", type=float, required=True, help="Copy only rows with prob_real strictly above this value.")
    return parser.parse_args()


def main():
    args = parse_args()
    scores_path = Path(args.scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    with scores_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prob_text = row.get("prob_real", "")
            if row.get("error") or not prob_text:
                skipped += 1
                continue
            if float(prob_text) <= args.min_prob:
                skipped += 1
                continue

            source = Path(row["original_path"])
            destination = output_dir / source.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
            shutil.copy2(source, destination)
            copied += 1

    print(f"copied={copied} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()

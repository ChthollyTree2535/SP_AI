import argparse
import csv
import re
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from train_transientx import TransientXDataset, build_model


PROB_PREFIX_RE = re.compile(r"^\d\.\d{8}_")


def parse_args():
    parser = argparse.ArgumentParser(description="Score TransientX PNG candidates with a trained model.")
    parser.add_argument("--model", required=True, help="Path to best_model.pt.")
    parser.add_argument("--input", required=True, help="PNG file or directory containing PNG files.")
    parser.add_argument("--output", default="scores.csv", help="Output CSV path.")
    parser.add_argument(
        "--renamed-output-dir",
        default=None,
        help=(
            "Optional directory for copied PNGs renamed as '<probability>_<original name>'. "
            "The input directory tree is preserved."
        ),
    )
    parser.add_argument(
        "--move-output-dir",
        default=None,
        help=(
            "Optional directory for MOVED PNGs renamed as '<probability>_<original name>'. "
            "This changes the original input tree. If a relative name such as 'plot' is used, "
            "it is created under the input directory."
        ),
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="When copying or moving renamed PNGs, put all files directly in the output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Include files that already look probability-prefixed and files under plot directories.",
    )
    parser.add_argument(
        "--min-copy-prob",
        type=float,
        default=0.0,
        help="Only copy renamed PNGs with prob_real >= this value. Default copies all scored PNGs.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes for PNG decode/resize. Use 2-6 on Windows for large batches.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision for faster inference and lower GPU memory use.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def is_probability_prefixed(path):
    return bool(PROB_PREFIX_RE.match(Path(path).name))


def collect_pngs(path, force=False):
    path = Path(path)
    if path.is_file():
        return [path.resolve()]
    if not path.exists():
        raise FileNotFoundError(path)
    root = path.resolve()
    pngs = []
    for png in sorted(root.rglob("*.png")):
        resolved = png.resolve()
        rel = safe_relative_path(resolved, root)
        if not force and "plot" in rel.parts:
            continue
        if not force and is_probability_prefixed(resolved):
            continue
        pngs.append(resolved)
    return pngs


def safe_relative_path(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def unique_destination(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(1, 10000):
        candidate = parent / f"{stem}__{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free filename for {path}")


def copy_renamed_images(rows, input_path, output_dir, min_prob):
    input_path = Path(input_path).resolve()
    input_root = input_path.parent if input_path.is_file() else input_path
    output_dir = Path(output_dir).resolve()
    copied = 0
    for row in rows:
        src = Path(row["path"]).resolve()
        prob = row["prob_real"]
        rel = safe_relative_path(src, input_root)
        row["relative_path"] = str(rel)
        row["renamed_path"] = ""
        if prob < min_prob:
            continue
        dest_dir = output_dir / rel.parent
        if row.get("flatten"):
            dest_dir = output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{prob:.8f}_{src.name}"
        dest = unique_destination(dest_dir / dest_name)
        shutil.copy2(src, dest)
        row["renamed_path"] = str(dest)
        copied += 1
    return copied, output_dir


def resolve_output_dir(input_path, output_dir):
    output_dir = Path(output_dir)
    if output_dir.is_absolute():
        return output_dir.resolve()
    input_path = Path(input_path).resolve()
    base = input_path.parent if input_path.is_file() else input_path
    return (base / output_dir).resolve()


def move_renamed_images(rows, input_path, output_dir, min_prob, flatten=False, exclusive_min_prob=False):
    input_path = Path(input_path).resolve()
    input_root = input_path.parent if input_path.is_file() else input_path
    output_dir = resolve_output_dir(input_path, output_dir)
    moved = 0
    for row in rows:
        src = Path(row["original_path"]).resolve()
        prob = row["prob_real"]
        rel = safe_relative_path(src, input_root)
        row["relative_path"] = str(rel)
        row["moved_path"] = ""
        row["renamed_path"] = ""
        if prob < min_prob or (exclusive_min_prob and prob <= min_prob):
            continue
        dest_dir = output_dir if flatten else output_dir / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        original_name = src.name
        if is_probability_prefixed(src):
            dest_name = original_name
        else:
            dest_name = f"{prob:.8f}_{original_name}"
        dest = unique_destination(dest_dir / dest_name)
        shutil.move(str(src), str(dest))
        row["moved_path"] = str(dest)
        row["renamed_path"] = str(dest)
        moved += 1
    return moved, output_dir


def save_group_summary(path, rows):
    groups = {}
    for row in rows:
        rel = Path(row.get("relative_path") or Path(row["path"]).name)
        group = str(rel.parent) if str(rel.parent) != "." else "."
        entry = groups.setdefault(
            group,
            {
                "group": group,
                "count": 0,
                "pred_positive": 0,
                "max_prob_real": 0.0,
                "top_image": "",
                "top_output_path": "",
            },
        )
        entry["count"] += 1
        if row["pred_label"] == 1:
            entry["pred_positive"] += 1
        if row["prob_real"] > entry["max_prob_real"]:
            entry["max_prob_real"] = row["prob_real"]
            entry["top_image"] = row.get("original_path", row.get("path", ""))
            entry["top_output_path"] = row.get("moved_path") or row.get("renamed_path", "")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "count",
                "pred_positive",
                "max_prob_real",
                "top_image",
                "top_output_path",
            ],
        )
        writer.writeheader()
        for entry in sorted(groups.values(), key=lambda item: item["max_prob_real"], reverse=True):
            entry = dict(entry)
            entry["max_prob_real"] = f"{entry['max_prob_real']:.8f}"
            writer.writerow(entry)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("config", {}).get("args", {})
    crop = train_args.get("crop", [68, 184, 647, 950])
    size = (int(train_args.get("height", 96)), int(train_args.get("width", 72)))
    model = build_model(
        train_args.get("model_arch", "small_resnet"),
        train_args.get("base_channels", 24),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, crop, size


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


class SafeTransientXDataset(Dataset):
    def __init__(self, items, crop, size):
        self.items = items
        self.dataset = TransientXDataset(items, crop=crop, size=size, train=False, crop_jitter=0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]["path"]
        try:
            image, _, loaded_path = self.dataset[idx]
            return {"image": image, "path": loaded_path, "error": ""}
        except Exception as exc:
            return {"image": None, "path": path, "error": f"{type(exc).__name__}: {exc}"}


def safe_collate(batch):
    images = []
    paths = []
    errors = []
    for item in batch:
        if item["image"] is None:
            errors.append({"path": item["path"], "error": item["error"]})
        else:
            images.append(item["image"])
            paths.append(item["path"])
    image_batch = torch.stack(images, dim=0) if images else None
    return image_batch, paths, errors


def format_progress_bar(done, total, width=28):
    total = max(1, int(total))
    done = min(int(done), total)
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def score_pngs(
    model,
    device,
    pngs,
    crop,
    size,
    batch_size,
    threshold,
    return_errors=False,
    num_workers=0,
    amp=False,
    progress_label=None,
):
    items = [{"path": str(p), "label": 0} for p in pngs]
    dataset = SafeTransientXDataset(items, crop=crop, size=size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        collate_fn=safe_collate,
    )

    rows = []
    errors = []
    use_amp = bool(amp and device.type == "cuda")
    processed = 0
    total = len(dataset)
    with torch.inference_mode():
        for images, paths, batch_errors in loader:
            processed += len(paths) + len(batch_errors)
            errors.extend(batch_errors)
            if images is None:
                if progress_label:
                    bar = format_progress_bar(processed, total)
                    print(
                        f"\r  scoring {progress_label} {bar} {processed}/{total} "
                        f"{processed / max(1, total):.1%}",
                        end="",
                        flush=True,
                    )
                continue
            images = images.to(device, non_blocking=True)
            amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with amp_context:
                logits = model(images)
            probs = torch.sigmoid(logits.float()).cpu().numpy().tolist()
            for path, prob in zip(paths, probs):
                rows.append(
                    {
                        "original_path": path,
                        "path": path,
                        "relative_path": "",
                        "renamed_path": "",
                        "moved_path": "",
                        "prob_real": float(prob),
                        "pred_label": int(float(prob) >= threshold),
                    }
                )
            if progress_label:
                bar = format_progress_bar(processed, total)
                print(
                    f"\r  scoring {progress_label} {bar} {processed}/{total} "
                    f"{processed / max(1, total):.1%}",
                    end="",
                    flush=True,
                )
    if progress_label:
        if processed < total:
            bar = format_progress_bar(total, total)
            print(f"\r  scoring {progress_label} {bar} {total}/{total} 100.0%", flush=True)
        else:
            print("", flush=True)

    if errors:
        print(f"  skipped_unreadable_pngs={len(errors)}", file=sys.stderr, flush=True)
    rows.sort(key=lambda row: row["prob_real"], reverse=True)
    if return_errors:
        return rows, errors
    return rows


def save_scores(output_path, rows, threshold):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "original_path",
                "path",
                "relative_path",
                "renamed_path",
                "moved_path",
                "prob_real",
                "pred_label",
                "percentile",
            ],
        )
        writer.writeheader()
        n = len(rows)
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "original_path": row.get("original_path", row.get("path", "")),
                    "path": row.get("path", ""),
                    "relative_path": row.get("relative_path", ""),
                    "renamed_path": row.get("renamed_path", ""),
                    "moved_path": row.get("moved_path", ""),
                    "prob_real": f"{row['prob_real']:.8f}",
                    "pred_label": int(row.get("pred_label", row["prob_real"] >= threshold)),
                    "percentile": f"{rank / max(1, n):.8f}",
                }
            )


def save_read_errors(output_path, errors):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "error"])
        writer.writeheader()
        for error in errors:
            writer.writerow(error)


def main():
    args = parse_args()
    if args.renamed_output_dir and args.move_output_dir:
        raise ValueError("Use either --renamed-output-dir or --move-output-dir, not both.")

    device = resolve_device(args.device, cpu=args.cpu)
    model, crop, size = load_model(args.model, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(
            f"device={device} gpu={torch.cuda.get_device_name(device)} "
            f"batch_size={args.batch_size} num_workers={args.num_workers} amp={args.amp}",
            flush=True,
        )
    else:
        print(
            f"device={device} batch_size={args.batch_size} num_workers={args.num_workers}",
            flush=True,
        )

    pngs = collect_pngs(args.input, force=args.force)
    if not pngs:
        raise RuntimeError(f"No PNG files found in {args.input}")

    rows, read_errors = score_pngs(
        model,
        device,
        pngs,
        crop,
        size,
        args.batch_size,
        args.threshold,
        return_errors=True,
        num_workers=args.num_workers,
        amp=args.amp,
        progress_label=Path(args.input).name,
    )
    input_path = Path(args.input).resolve()
    input_root = input_path.parent if input_path.is_file() else input_path
    for row in rows:
        row["relative_path"] = str(safe_relative_path(Path(row["original_path"]), input_root))
        row["renamed_path"] = ""
        row["moved_path"] = ""

    if args.renamed_output_dir:
        for row in rows:
            row["flatten"] = args.flatten
        copied, renamed_dir = copy_renamed_images(
            rows, args.input, args.renamed_output_dir, args.min_copy_prob
        )
        moved, moved_dir = 0, None
    elif args.move_output_dir:
        moved, moved_dir = move_renamed_images(
            rows,
            args.input,
            args.move_output_dir,
            args.min_copy_prob,
            flatten=args.flatten,
        )
        copied, renamed_dir = 0, None
    else:
        copied, renamed_dir = 0, None
        moved, moved_dir = 0, None

    save_scores(args.output, rows, args.threshold)
    group_summary_path = Path(args.output).with_name(Path(args.output).stem + "_group_summary.csv")
    save_group_summary(group_summary_path, rows)
    read_errors_path = Path(args.output).with_name(Path(args.output).stem + "_read_errors.csv")
    if read_errors:
        save_read_errors(read_errors_path, read_errors)

    message = f"scored={len(rows)} output={Path(args.output).resolve()}"
    message += f" group_summary={group_summary_path.resolve()}"
    if read_errors:
        message += f" skipped={len(read_errors)} read_errors={read_errors_path.resolve()}"
    if renamed_dir:
        message += f" copied={copied} renamed_output_dir={renamed_dir}"
    if moved_dir:
        message += f" moved={moved} move_output_dir={moved_dir}"
    print(message)


if __name__ == "__main__":
    main()

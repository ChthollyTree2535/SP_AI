from pathlib import Path

import torch

from predict_transientx import load_model, resolve_device


def main():
    model_path = Path("models/best_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    device = resolve_device("auto")
    model, crop, size = load_model(model_path, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OK: torch={torch.__version__}")
    print(f"OK: device={device}")
    print(f"OK: model={model_path} params={n_params}")
    print(f"OK: crop={crop} input_size={size[1]}x{size[0]}")


if __name__ == "__main__":
    main()

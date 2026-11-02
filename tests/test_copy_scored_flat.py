import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "copy_scored_flat.py"


class CopyScoredFlatTests(unittest.TestCase):
    def test_copies_only_strictly_above_threshold_with_original_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            output_dir = root / "flat"
            source_dir.mkdir()
            selected = source_dir / "selected.png"
            cutoff = source_dir / "cutoff.png"
            rejected = source_dir / "rejected.png"
            selected.write_bytes(b"selected")
            cutoff.write_bytes(b"cutoff")
            rejected.write_bytes(b"rejected")

            scores = root / "scores.csv"
            with scores.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["original_path", "prob_real", "error"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"original_path": str(selected), "prob_real": "0.80000001", "error": ""},
                        {"original_path": str(cutoff), "prob_real": "0.80000000", "error": ""},
                        {"original_path": str(rejected), "prob_real": "0.79999999", "error": ""},
                    ]
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--scores",
                    str(scores),
                    "--output-dir",
                    str(output_dir),
                    "--min-prob",
                    "0.8",
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(path.name for path in output_dir.iterdir()), ["selected.png"])
            self.assertEqual((output_dir / "selected.png").read_bytes(), b"selected")


if __name__ == "__main__":
    unittest.main()

"""Summarize eight independently trained MA fused models."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedproxy.ma.homogeneous import main


if __name__ == "__main__":
    main(["summarize", *sys.argv[1:]])

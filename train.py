﻿from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wetlands_ml_geoai.train import main


if __name__ == "__main__":
    main()

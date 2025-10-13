﻿import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wetlands_ml_geoai.sentinel2.cli import main


if __name__ == "__main__":
    main()

"""Thin launcher so `python main.py [image]` still works.
The real entrypoint lives in app.main()."""
import core._threadlimit  # noqa: F401 — first import: pin OpenBLAS before numpy loads
from app import main

if __name__ == "__main__":
    main()
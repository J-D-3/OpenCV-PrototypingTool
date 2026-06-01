"""Thin launcher so `python main.py [image]` still works.
The real entrypoint lives in app.main()."""
from app import main

if __name__ == "__main__":
    main()
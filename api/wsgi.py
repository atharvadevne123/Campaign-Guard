"""Gunicorn WSGI entry point — loads models once at worker startup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app, _load_models

_load_models()

if __name__ == "__main__":
    app.run()

"""Vercel entry point.

Vercel serves whatever WSGI callable named `app` this module exposes. The real
application lives in `app.py` at the repository root so that `python app.py`
keeps working locally without knowing anything about Vercel.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402

__all__ = ["app"]

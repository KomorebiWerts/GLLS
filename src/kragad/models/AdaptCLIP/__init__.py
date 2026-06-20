"""Compatibility exports for GLLS's AdaptCLIP localizer."""

from pathlib import Path
import sys

_package_dir = Path(__file__).resolve().parent
if str(_package_dir) not in sys.path:
    sys.path.insert(0, str(_package_dir))

from . import adaptcliplib
from .adaptcliplib import PQAdapter, TextualAdapter, VisualAdapter, available_models, fusion_fun, load
from .tools import get_transform

__all__ = [
    "adaptcliplib",
    "PQAdapter",
    "TextualAdapter",
    "VisualAdapter",
    "available_models",
    "fusion_fun",
    "get_transform",
    "load",
]

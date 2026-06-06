"""
Utility functions: path management and directory helpers.
All paths are derived from this file's location so scripts work
from any working directory (including inside Docker at /app).
"""

import os
import sys


def get_project_root() -> str:
    """Return the absolute path to the project root (one level above src/)."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(src_dir)


def get_data_dir() -> str:
    """Return the absolute path to the data/ directory."""
    return os.path.join(get_project_root(), "data")


def get_results_dir(question: str) -> str:
    """
    Return the absolute path to results/<question>/ and create it if needed.
    question: 'q1' or 'q2'
    """
    path = os.path.join(get_project_root(), "results", question)
    os.makedirs(path, exist_ok=True)
    return path


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def add_src_to_path() -> None:
    """Add src/ to sys.path so sibling modules can be imported directly."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

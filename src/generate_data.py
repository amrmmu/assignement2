"""
Generate synthetic vaccination datasets for Malaysia.

People  : 10 000 vaccinees  → data/people.csv
Centres :    100 vaccination centres → data/centre.csv

Coordinates are drawn uniformly within Peninsular Malaysia
(lat 1.2–6.7 °N, lon 99.6–104.7 °E).  A fixed random seed
guarantees reproducibility.

Run:
    python src/generate_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_data_dir, ensure_dir


# Peninsular Malaysia bounding box (approximate)
LAT_MIN, LAT_MAX = 1.2, 6.7
LON_MIN, LON_MAX = 99.6, 104.7


def _random_coords(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lats = rng.uniform(LAT_MIN, LAT_MAX, n)
    lons = rng.uniform(LON_MIN, LON_MAX, n)
    return lats, lons


def generate_people(
    n: int = 10_000,
    output_path: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate vaccinee dataset and write to CSV."""
    if output_path is None:
        output_path = os.path.join(get_data_dir(), "people.csv")

    if os.path.exists(output_path) and not overwrite:
        print(f"[generate_data] people.csv already exists at {output_path} — skipping.")
        return pd.read_csv(output_path)

    ensure_dir(os.path.dirname(output_path))
    lats, lons = _random_coords(n, seed=42)
    df = pd.DataFrame(
        {
            "person_id": [f"Person_{i + 1}" for i in range(n)],
            "latitude": lats,
            "longitude": lons,
        }
    )
    df.to_csv(output_path, index=False)
    print(f"[generate_data] Generated {n:,} people -> {output_path}")
    return df


def generate_centres(
    n: int = 100,
    output_path: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate vaccination centre dataset and write to CSV."""
    if output_path is None:
        output_path = os.path.join(get_data_dir(), "centre.csv")

    if os.path.exists(output_path) and not overwrite:
        print(f"[generate_data] centre.csv already exists at {output_path} — skipping.")
        return pd.read_csv(output_path)

    ensure_dir(os.path.dirname(output_path))
    lats, lons = _random_coords(n, seed=123)
    df = pd.DataFrame(
        {
            "centre_id": [f"Centre_{i + 1}" for i in range(n)],
            "latitude": lats,
            "longitude": lons,
        }
    )
    df.to_csv(output_path, index=False)
    print(f"[generate_data] Generated {n:,} centres -> {output_path}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic vaccination data")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    generate_people(overwrite=args.overwrite)
    generate_centres(overwrite=args.overwrite)
    print("[generate_data] Done.")

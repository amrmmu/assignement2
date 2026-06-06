"""
CSV data loading with automatic column detection and ID generation.
Supports canonical filenames and common fallback names.
Detects latitude/longitude columns by common names.
Generates person_id / centre_id when no ID column is present.
"""

import os
import pandas as pd
from typing import Tuple


# ── column name registries ────────────────────────────────────────────────────

_LAT_NAMES = {"lat", "latitude"}
_LON_NAMES = {"lon", "lng", "longitude", "long"}

_PERSON_ID_NAMES = {"person_id", "id", "person", "people", "people_id", "vaccinee_id", "no", "index"}
_CENTRE_ID_NAMES = {"centre_id", "center_id", "id", "centre", "center", "ppv", "no", "index"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_file(primary: str, fallbacks: list[str], data_dir: str) -> str:
    """Return the first existing file path among primary + fallbacks."""
    for name in [primary] + fallbacks:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    checked = ", ".join([primary] + fallbacks)
    raise FileNotFoundError(
        f"Could not find any of [{checked}] in directory '{data_dir}'. "
        "Please place the CSV files in the data/ folder."
    )


def _detect_column(df: pd.DataFrame, name_set: set[str], column_role: str) -> str:
    """
    Return the first DataFrame column whose lowercase name is in name_set.
    Raises ValueError with a helpful message if none found.
    """
    for col in df.columns:
        if col.strip().lower() in name_set:
            return col
    raise ValueError(
        f"Could not detect {column_role} column. "
        f"Expected one of {sorted(name_set)}. "
        f"Found columns: {df.columns.tolist()}"
    )


def _detect_id_column(df: pd.DataFrame, name_set: set[str]) -> str | None:
    """Return the first matching ID column, or None if not found."""
    for col in df.columns:
        if col.strip().lower() in name_set:
            return col
    return None


# ── public loaders ────────────────────────────────────────────────────────────

def load_people(data_dir: str) -> pd.DataFrame:
    """
    Load the vaccinee dataset.

    Returns a DataFrame with columns:
        person_id  (str)
        lat        (float)
        lon        (float)
    """
    path = _find_file(
        "people.csv",
        ["people(in).csv", "People.csv", "People(in).csv"],
        data_dir,
    )
    raw = pd.read_csv(path)

    lat_col = _detect_column(raw, _LAT_NAMES, "latitude")
    lon_col = _detect_column(raw, _LON_NAMES, "longitude")
    id_col = _detect_id_column(raw, _PERSON_ID_NAMES)

    result = pd.DataFrame()
    if id_col:
        result["person_id"] = raw[id_col].astype(str)
    else:
        result["person_id"] = [f"Person_{i + 1}" for i in range(len(raw))]

    result["lat"] = raw[lat_col].astype(float)
    result["lon"] = raw[lon_col].astype(float)

    # Basic validation
    invalid = result[(result["lat"].abs() > 90) | (result["lon"].abs() > 180)]
    if not invalid.empty:
        raise ValueError(
            f"{len(invalid)} rows have out-of-range coordinates in {path}."
        )

    return result.reset_index(drop=True)


def load_centres(data_dir: str) -> pd.DataFrame:
    """
    Load the vaccination centre dataset.

    Returns a DataFrame with columns:
        centre_id  (str)
        lat        (float)
        lon        (float)
    """
    path = _find_file(
        "centre.csv",
        ["centre(in).csv", "center.csv", "center(in).csv",
         "Centre.csv", "Center.csv"],
        data_dir,
    )
    raw = pd.read_csv(path)

    lat_col = _detect_column(raw, _LAT_NAMES, "latitude")
    lon_col = _detect_column(raw, _LON_NAMES, "longitude")
    id_col = _detect_id_column(raw, _CENTRE_ID_NAMES)

    result = pd.DataFrame()
    if id_col:
        result["centre_id"] = raw[id_col].astype(str)
    else:
        result["centre_id"] = [f"Centre_{i + 1}" for i in range(len(raw))]

    result["lat"] = raw[lat_col].astype(float)
    result["lon"] = raw[lon_col].astype(float)

    invalid = result[(result["lat"].abs() > 90) | (result["lon"].abs() > 180)]
    if not invalid.empty:
        raise ValueError(
            f"{len(invalid)} rows have out-of-range coordinates in {path}."
        )

    return result.reset_index(drop=True)


def load_both(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: returns (people_df, centres_df)."""
    return load_people(data_dir), load_centres(data_dir)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import get_data_dir

    p, c = load_both(get_data_dir())
    print(f"People:  {len(p)} rows  | columns: {p.columns.tolist()}")
    print(f"Centres: {len(c)} rows  | columns: {c.columns.tolist()}")
    print(p.head(3))
    print(c.head(3))

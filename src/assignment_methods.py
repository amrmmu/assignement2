"""
Three substantially different methods for assigning vaccinees to
the nearest vaccination centre.

Method 1  – Pure Python nested loops (math module, no NumPy)
Method 1P – Parallel version of Method 1 using multiprocessing.Pool
            (used for Question 1 so that CPU-count limits have a real effect)
Method 2  – Vectorised NumPy broadcasting (full distance matrix)
Method 3  – Spatial indexing with scipy.spatial.cKDTree

Each function accepts:
    people_df  : DataFrame with columns [person_id, lat, lon]
    centres_df : DataFrame with columns [centre_id, lat, lon]

Each function returns a DataFrame with columns:
    person_id, person_lat, person_lon,
    assigned_centre_id, centre_lat, centre_lon,
    distance_km, method_name
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Allow running this module standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distance import (
    haversine_python,
    haversine_numpy_matrix,
    latlon_to_cartesian,
    chord_to_haversine_km,
)

# ── shared output column order ────────────────────────────────────────────────

_OUTPUT_COLS = [
    "person_id", "person_lat", "person_lon",
    "assigned_centre_id", "centre_lat", "centre_lon",
    "distance_km", "method_name",
]


# ── cgroup-aware CPU count ────────────────────────────────────────────────────

def _get_container_cpu_count() -> int:
    """
    Return the number of CPUs available to this container.

    Reads the Docker/cgroup CPU quota so that a container limited to 1 CPU
    returns 1 and a container limited to 4 CPUs returns 4 — regardless of
    how many physical cores the host machine has.

    Falls back to os.cpu_count() when not running inside a container
    (e.g. local development on Windows/Mac).
    """
    # cgroup v1 (Docker default on older kernels)
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0:
            return max(1, round(quota / period))
    except Exception:
        pass

    # cgroup v2 (modern kernels / Docker Desktop)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts[0] != "max":
            return max(1, round(int(parts[0]) / int(parts[1])))
    except Exception:
        pass

    return max(1, os.cpu_count() or 1)


# ── Method 1: Pure Python nested loops ───────────────────────────────────────

def method1_python_loops(
    people_df: pd.DataFrame,
    centres_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each vaccinee to the nearest centre using a plain Python
    nested loop.  Distance is computed with the math module only —
    no NumPy, no vectorisation.

    Complexity: O(n_people × n_centres)
    """
    centres_records = centres_df.to_dict("records")
    rows = []

    for person in people_df.itertuples(index=False):
        best_dist = float("inf")
        best_centre = None

        for centre in centres_records:
            dist = haversine_python(
                person.lat, person.lon,
                centre["lat"], centre["lon"],
            )
            if dist < best_dist:
                best_dist = dist
                best_centre = centre

        rows.append(
            (
                person.person_id,
                person.lat,
                person.lon,
                best_centre["centre_id"],
                best_centre["lat"],
                best_centre["lon"],
                best_dist,
                "Method1_PythonLoops",
            )
        )

    return pd.DataFrame(rows, columns=_OUTPUT_COLS)


# ── Method 1 — Parallel (used for Question 1) ────────────────────────────────
#
# WHY THIS EXISTS:
#   Python's Global Interpreter Lock (GIL) prevents threads from running
#   Python bytecode in parallel, so a plain for-loop always uses exactly
#   one CPU regardless of how many are available.  multiprocessing.Pool
#   spawns *separate OS processes*, each with its own interpreter and no
#   shared GIL.  The 10,000 people are split into n_workers equal chunks;
#   each chunk runs on one process.
#
#   Effect on Docker resource limits:
#     computer1 (4 CPUs)  →  n_workers = 4  →  ~4× parallelism
#     computer2 (1 CPU)   →  n_workers = 1  →  sequential (same as Method 1)
#
#   This is what makes the CPU limit *actually matter* for Question 1.

def _worker_assign_chunk(args: tuple) -> list:
    """
    Module-level worker function — must be defined at module scope so
    Python's pickle can serialise it when sending work to child processes.

    Receives a (people_chunk DataFrame, centres_records list) tuple and
    returns a list of result row tuples.
    """
    people_chunk, centres_records = args
    rows = []
    for person in people_chunk.itertuples(index=False):
        best_dist = float("inf")
        best_centre = None
        for centre in centres_records:
            dist = haversine_python(
                person.lat, person.lon,
                centre["lat"], centre["lon"],
            )
            if dist < best_dist:
                best_dist = dist
                best_centre = centre
        rows.append((
            person.person_id, person.lat, person.lon,
            best_centre["centre_id"], best_centre["lat"], best_centre["lon"],
            best_dist, "Method1_ParallelLoops",
        ))
    return rows


def method1_python_loops_parallel(
    people_df: pd.DataFrame,
    centres_df: pd.DataFrame,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Parallel version of the pure-Python Haversine loop.

    Splits the 10,000 vaccinees into n_workers equal chunks and assigns
    each chunk to a separate OS process via multiprocessing.Pool.  Each
    process runs the identical single-core Haversine loop as Method 1.

    n_workers is automatically set from the container's cgroup CPU quota:
      - computer1 (cpus=4.0)  →  4 workers  →  ~4× faster than Method 1
      - computer2 (cpus=1.0)  →  1 worker   →  same speed as Method 1

    This means the Docker CPU limit directly controls execution time,
    making Question 1 produce a statistically measurable difference.
    """
    if n_workers is None:
        n_workers = _get_container_cpu_count()

    centres_records = centres_df.to_dict("records")

    # Split on integer index positions so each chunk is always a DataFrame
    idx_splits = np.array_split(np.arange(len(people_df)), n_workers)
    chunks = [people_df.iloc[idx].reset_index(drop=True) for idx in idx_splits if len(idx) > 0]

    with Pool(processes=n_workers) as pool:
        chunk_results = pool.map(
            _worker_assign_chunk,
            [(chunk, centres_records) for chunk in chunks],
        )

    all_rows = [row for chunk_rows in chunk_results for row in chunk_rows]
    return pd.DataFrame(all_rows, columns=_OUTPUT_COLS)


# ── Method 2: NumPy broadcasting ─────────────────────────────────────────────

def method2_numpy_broadcasting(
    people_df: pd.DataFrame,
    centres_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full (n_people × n_centres) Haversine distance matrix with
    NumPy broadcasting, then find the column-argmin for each row.

    No Python loop over individuals — the inner loop is expressed as
    array operations handled by NumPy in compiled C code.

    Complexity: O(n_people × n_centres) — same as Method 1 but far
    lower constant factor due to vectorisation.
    """
    p_lats = people_df["lat"].values
    p_lons = people_df["lon"].values
    c_lats = centres_df["lat"].values
    c_lons = centres_df["lon"].values

    # Full distance matrix: shape (n_people, n_centres)
    dist_matrix = haversine_numpy_matrix(p_lats, p_lons, c_lats, c_lons)

    # Index of nearest centre for each person
    nearest_idx = np.argmin(dist_matrix, axis=1)
    min_dists = dist_matrix[np.arange(len(people_df)), nearest_idx]

    return pd.DataFrame(
        {
            "person_id": people_df["person_id"].values,
            "person_lat": p_lats,
            "person_lon": p_lons,
            "assigned_centre_id": centres_df["centre_id"].values[nearest_idx],
            "centre_lat": c_lats[nearest_idx],
            "centre_lon": c_lons[nearest_idx],
            "distance_km": min_dists,
            "method_name": "Method2_NumpyBroadcast",
        }
    )[_OUTPUT_COLS]


# ── Method 3: Spatial indexing with cKDTree ───────────────────────────────────

def method3_spatial_indexing(
    people_df: pd.DataFrame,
    centres_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert lat/lon to 3-D Cartesian coordinates on the unit sphere,
    build a cKDTree over the 100 centres, then query the nearest centre
    for all 10 000 vaccinees in one vectorised call.

    Why cKDTree beats a brute-force loop:
      • The KD-tree partitions the 3-D space into axis-aligned bounding
        boxes so each query visits O(log n_centres) nodes instead of all
        100 centres.
      • scipy's cKDTree.query() is implemented in Cython, offering
        near-C speed with no Python overhead per comparison.
      • The chord distance on the unit sphere is a monotone function of
        the great-circle distance, so nearest-chord == nearest-Haversine.

    The chord distances returned by cKDTree are converted back to km via
    the exact Haversine formula before being stored in the result.
    """
    p_lats = people_df["lat"].values
    p_lons = people_df["lon"].values
    c_lats = centres_df["lat"].values
    c_lons = centres_df["lon"].values

    # Project to unit sphere
    people_xyz = latlon_to_cartesian(p_lats, p_lons)
    centres_xyz = latlon_to_cartesian(c_lats, c_lons)

    # Build KD-tree once for all centres
    tree = cKDTree(centres_xyz)

    # Query: one call retrieves nearest centre for every person
    chord_dists, nearest_idx = tree.query(people_xyz, k=1, workers=1)

    # Convert chord → km
    dist_km = chord_to_haversine_km(chord_dists)

    return pd.DataFrame(
        {
            "person_id": people_df["person_id"].values,
            "person_lat": p_lats,
            "person_lon": p_lons,
            "assigned_centre_id": centres_df["centre_id"].values[nearest_idx],
            "centre_lat": c_lats[nearest_idx],
            "centre_lon": c_lons[nearest_idx],
            "distance_km": dist_km,
            "method_name": "Method3_SpatialIndex",
        }
    )[_OUTPUT_COLS]


# ── registry ──────────────────────────────────────────────────────────────────

METHODS = {
    "method1": ("Method1_PythonLoops",      method1_python_loops),
    "method2": ("Method2_NumpyBroadcast",   method2_numpy_broadcasting),
    "method3": ("Method3_SpatialIndex",     method3_spatial_indexing),
}


if __name__ == "__main__":
    from data_loader import load_both
    from utils import get_data_dir

    people_df, centres_df = load_both(get_data_dir())
    print(f"Loaded {len(people_df)} people and {len(centres_df)} centres.")

    for key, (name, fn) in METHODS.items():
        import time
        t0 = time.perf_counter()
        result = fn(people_df, centres_df)
        elapsed = time.perf_counter() - t0
        print(f"{name}: {elapsed:.4f}s  | result shape: {result.shape}")
        print(result.head(2).to_string(index=False))
        print()

    # Also test the parallel version
    import time
    n = _get_container_cpu_count()
    print(f"\nParallel version (n_workers={n}):")
    t0 = time.perf_counter()
    result_p = method1_python_loops_parallel(people_df, centres_df)
    elapsed = time.perf_counter() - t0
    print(f"Method1_ParallelLoops: {elapsed:.4f}s  | result shape: {result_p.shape}")


# Allow running this module standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distance import (
    haversine_python,
    haversine_numpy_matrix,
    latlon_to_cartesian,
    chord_to_haversine_km,
)

# ── shared output column order ────────────────────────────────────────────────

_OUTPUT_COLS = [
    "person_id", "person_lat", "person_lon",
    "assigned_centre_id", "centre_lat", "centre_lon",
    "distance_km", "method_name",
]


# ── Method 1: Pure Python nested loops ───────────────────────────────────────

def method1_python_loops(
    people_df: pd.DataFrame,
    centres_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each vaccinee to the nearest centre using a plain Python
    nested loop.  Distance is computed with the math module only —
    no NumPy, no vectorisation.

    Complexity: O(n_people × n_centres)
    """
    centres_records = centres_df.to_dict("records")
    rows = []

    for person in people_df.itertuples(index=False):
        best_dist = float("inf")
        best_centre = None

        for centre in centres_records:
            dist = haversine_python(
                person.lat, person.lon,
                centre["lat"], centre["lon"],
            )
            if dist < best_dist:
                best_dist = dist
                best_centre = centre

        rows.append(
            (
                person.person_id,
                person.lat,
                person.lon,
                best_centre["centre_id"],
                best_centre["lat"],
                best_centre["lon"],
                best_dist,
                "Method1_PythonLoops",
            )
        )

    return pd.DataFrame(rows, columns=_OUTPUT_COLS)


# ── Method 2: NumPy broadcasting ─────────────────────────────────────────────

def method2_numpy_broadcasting(
    people_df: pd.DataFrame,
    centres_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full (n_people × n_centres) Haversine distance matrix with
    NumPy broadcasting, then find the column-argmin for each row.

    No Python loop over individuals — the inner loop is expressed as
    array operations handled by NumPy in compiled C code.

    Complexity: O(n_people × n_centres) — same as Method 1 but far
    lower constant factor due to vectorisation.
    """
    p_lats = people_df["lat"].values
    p_lons = people_df["lon"].values
    c_lats = centres_df["lat"].values
    c_lons = centres_df["lon"].values

    # Full distance matrix: shape (n_people, n_centres)
    dist_matrix = haversine_numpy_matrix(p_lats, p_lons, c_lats, c_lons)

    # Index of nearest centre for each person
    nearest_idx = np.argmin(dist_matrix, axis=1)
    min_dists = dist_matrix[np.arange(len(people_df)), nearest_idx]

    return pd.DataFrame(
        {
            "person_id": people_df["person_id"].values,
            "person_lat": p_lats,
            "person_lon": p_lons,
            "assigned_centre_id": centres_df["centre_id"].values[nearest_idx],
            "centre_lat": c_lats[nearest_idx],
            "centre_lon": c_lons[nearest_idx],
            "distance_km": min_dists,
            "method_name": "Method2_NumpyBroadcast",
        }
    )[_OUTPUT_COLS]


# ── Method 3: Spatial indexing with cKDTree ───────────────────────────────────

def method3_spatial_indexing(
    people_df: pd.DataFrame,
    centres_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert lat/lon to 3-D Cartesian coordinates on the unit sphere,
    build a cKDTree over the 100 centres, then query the nearest centre
    for all 10 000 vaccinees in one vectorised call.

    Why cKDTree beats a brute-force loop:
      • The KD-tree partitions the 3-D space into axis-aligned bounding
        boxes so each query visits O(log n_centres) nodes instead of all
        100 centres.
      • scipy's cKDTree.query() is implemented in Cython, offering
        near-C speed with no Python overhead per comparison.
      • The chord distance on the unit sphere is a monotone function of
        the great-circle distance, so nearest-chord == nearest-Haversine.

    The chord distances returned by cKDTree are converted back to km via
    the exact Haversine formula before being stored in the result.
    """
    p_lats = people_df["lat"].values
    p_lons = people_df["lon"].values
    c_lats = centres_df["lat"].values
    c_lons = centres_df["lon"].values

    # Project to unit sphere
    people_xyz = latlon_to_cartesian(p_lats, p_lons)
    centres_xyz = latlon_to_cartesian(c_lats, c_lons)

    # Build KD-tree once for all centres
    tree = cKDTree(centres_xyz)

    # Query: one call retrieves nearest centre for every person
    chord_dists, nearest_idx = tree.query(people_xyz, k=1, workers=1)

    # Convert chord → km
    dist_km = chord_to_haversine_km(chord_dists)

    return pd.DataFrame(
        {
            "person_id": people_df["person_id"].values,
            "person_lat": p_lats,
            "person_lon": p_lons,
            "assigned_centre_id": centres_df["centre_id"].values[nearest_idx],
            "centre_lat": c_lats[nearest_idx],
            "centre_lon": c_lons[nearest_idx],
            "distance_km": dist_km,
            "method_name": "Method3_SpatialIndex",
        }
    )[_OUTPUT_COLS]


# ── registry ──────────────────────────────────────────────────────────────────

METHODS = {
    "method1": ("Method1_PythonLoops",      method1_python_loops),
    "method2": ("Method2_NumpyBroadcast",   method2_numpy_broadcasting),
    "method3": ("Method3_SpatialIndex",     method3_spatial_indexing),
}


if __name__ == "__main__":
    from data_loader import load_both
    from utils import get_data_dir

    people_df, centres_df = load_both(get_data_dir())
    print(f"Loaded {len(people_df)} people and {len(centres_df)} centres.")

    for key, (name, fn) in METHODS.items():
        import time
        t0 = time.perf_counter()
        result = fn(people_df, centres_df)
        elapsed = time.perf_counter() - t0
        print(f"{name}: {elapsed:.4f}s  | result shape: {result.shape}")
        print(result.head(2).to_string(index=False))
        print()

"""
Haversine distance functions.

Three implementations are provided:
  - haversine_python()        : scalar, pure Python + math
  - haversine_numpy_matrix()  : full (n_people × n_centres) distance matrix
  - chord_to_haversine_km()   : convert unit-sphere chord distance → km
                                (used after cKDTree nearest-neighbour lookup)
"""

import math
import numpy as np

EARTH_RADIUS_KM: float = 6371.0


# ── pure-Python scalar ─────────────────────────────────────────────────────

def haversine_python(lat1: float, lon1: float,
                     lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in kilometres between two points
    specified by decimal degrees.  Uses the math module only.
    """
    R = EARTH_RADIUS_KM
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2.0 * R * math.asin(math.sqrt(a))


# ── NumPy vectorised matrix ────────────────────────────────────────────────

def haversine_numpy_matrix(
    people_lats: np.ndarray,
    people_lons: np.ndarray,
    centre_lats: np.ndarray,
    centre_lons: np.ndarray,
) -> np.ndarray:
    """
    Return a (n_people × n_centres) distance matrix (km) computed with
    NumPy broadcasting.  All four inputs are 1-D float arrays.
    """
    R = EARTH_RADIUS_KM

    # Reshape for broadcasting: people → (n, 1), centres → (1, m)
    p_lat = np.radians(people_lats[:, np.newaxis])
    p_lon = np.radians(people_lons[:, np.newaxis])
    c_lat = np.radians(centre_lats[np.newaxis, :])
    c_lon = np.radians(centre_lons[np.newaxis, :])

    dphi = c_lat - p_lat
    dlam = c_lon - p_lon

    a = (np.sin(dphi / 2) ** 2
         + np.cos(p_lat) * np.cos(c_lat) * np.sin(dlam / 2) ** 2)

    # Clip to [0, 1] guards against tiny floating-point overshoots
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ── chord-distance → Haversine km ─────────────────────────────────────────

def latlon_to_cartesian(lat_deg: np.ndarray,
                        lon_deg: np.ndarray) -> np.ndarray:
    """
    Convert arrays of (lat, lon) in decimal degrees to unit-sphere 3-D
    Cartesian coordinates.  Returns an (n, 3) array of (x, y, z).
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])


def chord_to_haversine_km(chord_dist: np.ndarray) -> np.ndarray:
    """
    Convert Euclidean chord distance(s) on the unit sphere to
    great-circle distance in kilometres.

    Derivation:
        chord = 2 sin(θ/2)  →  θ = 2 arcsin(chord/2)
        distance_km = R · θ
    """
    ratio = np.clip(chord_dist / 2.0, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(ratio)

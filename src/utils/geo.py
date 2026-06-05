"""Geospatial utility functions."""

import math
import numpy as np


def haversine(lat1, lng1, lat2, lng2):
    """Jarak meter antara 2 lat/lng."""
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def haversine_vectorized(lat1, lng1, lats, lngs):
    """Vectorized haversine pakai numpy."""
    R = 6371000
    p1 = math.radians(lat1)
    p2 = np.radians(lats)
    dp = np.radians(lats - lat1)
    dl = np.radians(lngs - lng1)
    a = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
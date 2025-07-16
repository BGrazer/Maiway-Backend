import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from ..utils.geo_utils import haversine_distance
import requests
try:
    import polyline as _poly
except ImportError:  # graceful degradation if polyline not installed
    _poly = None

# ------------------------------------------------------
#  Configuration constants – tweak via env if necessary
# ------------------------------------------------------
TRIKE_CATCHMENT_KM = 0.7   # user must be within 400 m of a terminal

# The ride should be *useful*: at least 0.5 km long (else walking is faster)
# but we still cap it at 2 km so the segment stays short-haul.

TRIKE_MIN_DISTANCE_KM = 0.1   # min ride distance to boarding stop
TRIKE_MAX_DISTANCE_KM = 2.0   # max ride distance to boarding stop
TRIKE_FLAT_FARE = 21.0       # PHP flat fare (₱16 base → ₱21 total)
TRIKE_SPEED_KMPH = 30.0      # assumed average speed on local roads

# Path inside the backend data dir – can be overridden via env
DEFAULT_GEOJSON_PATH = os.getenv("TRICYCLE_TERMINALS_GEOJSON", "routing_data/tricycle_terminals.geojson")

MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN')


def _mapbox_directions(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[List[List[float]]]:
    """Return a list of [lon, lat] pairs from Mapbox Directions API or None."""
    if not MAPBOX_TOKEN or _poly is None:
        return None
    try:
        url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{lon1},{lat1};{lon2},{lat2}"
        resp = requests.get(url, params={
            'geometries': 'polyline6',
            'overview': 'full',
            'access_token': MAPBOX_TOKEN,
        }, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        routes = data.get('routes')
        if not routes:
            return None
        geom = routes[0].get('geometry')
        if not geom:
            return None
        coords = _poly.decode(geom, precision=6)
        # decode returns (lat, lon)
        return [[lon, lat] for lat, lon in coords]
    except Exception:
        return None


def load_trike_terminals(geojson_path: str = DEFAULT_GEOJSON_PATH) -> List[Dict]:
    """Return a list of simple dicts with id/lat/lon for each terminal.

    The GeoJSON must be a FeatureCollection with Point geometries.
    If the file is missing or malformed, an empty list is returned so that the
    caller can silently skip tricycle logic.
    """
    # Allow for backwards-compat: if the expected filename is missing try
    # a common alternate (older) name so the feature still works out of
    # the box.

    path = Path(geojson_path)
    if not path.exists():
        alt_path = Path("routing_data/tricycle.geojson")
        if alt_path.exists():
            path = alt_path
        else:
            return []
    try:
        with path.open("r", encoding="utf-8") as f:
            geo = json.load(f)
        out: List[Dict] = []
        for feat in geo.get("features", []):
            geom = feat.get("geometry", {})
            if geom.get("type") != "Point":
                continue
            lon, lat = geom.get("coordinates", [None, None])
            if lat is None or lon is None:
                continue
            props = feat.get("properties", {}) or {}
            tid = props.get("terminal_id") or props.get("id") or f"TERM_{len(out)+1}"
            out.append({
                "id": str(tid),
                "lat": float(lat),
                "lon": float(lon),
                "name": props.get("name", f"Trike Terminal {tid}")
            })
        return out
    except Exception:
        return []


def nearest_terminal(lat: float, lon: float, terminals: List[Dict], max_km: float = TRIKE_CATCHMENT_KM) -> Optional[Tuple[Dict, float]]:
    """Return the closest terminal within ``max_km`` else None."""
    best: Optional[Tuple[Dict, float]] = None
    for t in terminals:
        d = haversine_distance(lat, lon, t["lat"], t["lon"])
        if d <= max_km and (best is None or d < best[1]):
            best = (t, d)
    return best


def build_trike_segment(origin: Dict, dest: Dict) -> Dict:
    """Create a route segment dict for a tricycle ride.

    ``origin`` / ``dest`` must contain id, name, lat, lon keys in the format the
    wider MaiWay engine already uses for walking segments.  Polyline is *not*
    generated here—callers are expected to fill it in (e.g., via Mapbox) and set
    ``polyline_source`` accordingly.  That keeps this helper self-contained and
    free of external HTTP dependencies.
    """
    kms = haversine_distance(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
    # Guard – callers should pass only feasible pairs, but we sanity-check
    if kms > TRIKE_MAX_DISTANCE_KM:
        raise ValueError("Tricycle segment exceeds configured max distance")

    seg = {
        "from_stop": origin,
        "to_stop": dest,
        "mode": "Tricycle",
        "distance": round(kms, 3),
        "fare": TRIKE_FLAT_FARE,
        "trip_id": "TRICYCLE",
        "route_id": "TRICYCLE",
        "reason": "first_mile",
        "polyline": [],
        "polyline_source": "",
    }

    poly = _mapbox_directions(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
    if poly:
        seg["polyline"] = poly
        seg["polyline_source"] = "mapbox_driving"
    else:
        seg["polyline"] = [[origin["lon"], origin["lat"]], [dest["lon"], dest["lat"]]]
        seg["polyline_source"] = "straight_line"

    return seg 
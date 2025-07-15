"""
Unified Route Service: Combined routing service with GTFS compliance and logical routing
Combines features from RouteService and ImprovedRouteService
Now with full Manila OSMnx graph for better A* performance
"""

import math
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import networkx as nx
import pandas as pd
import logging
import sys
# import osmnx as ox  # Commented out to fix SSL error
import requests
import json
import heapq
import numpy as np
from datetime import datetime, timedelta
from .core_shape_generator import CoreShapeGenerator
from .config import config
from .models.route_segments import TransitSegment, WalkingSegment, RouteSegment, CompleteRoute
from maiwayrouting.utils.geo_utils import haversine_distance
from .utils.fare_utils import calculate_fare
from .graph.graph_builder import build_transit_graph, build_smart_walking_edges, build_complete_graph, is_useful_transfer
from .routing.algorithms import find_route_astar
from .networkx_cost_functions import make_cost_function
from .utils.trike_utils import (
    load_trike_terminals,
    nearest_terminal,
    build_trike_segment,
    TRIKE_CATCHMENT_KM,
    TRIKE_MIN_DISTANCE_KM,
    TRIKE_MAX_DISTANCE_KM,
    TRIKE_FLAT_FARE,
)


class UnifiedRouteService:
    """
    Unified route service that provides logical routes by:
    1. Using proper GTFS trip segments (consecutive stops only)
    2. Adding smart walking connections for transfers
    3. Limiting walking distance and transfers
    4. Prioritizing transit over walking
    5. Generates multiple candidate routes, prunes, deduplicates, and outputs Sakay-like alternatives
    6. Provides high-level API for coordinate-based routing with walking segments
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the unified route service"""
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # GTFS data storage
        self.stops = {}
        self.routes = {}
        self.trips = {}
        self.stop_times = []
        self.shapes = {}  # Cache for GTFS shapes
        
        # Graph structures
        self.transit_graph = nx.DiGraph()  # Transit-only graph
        self.walking_graph = nx.DiGraph()  # Walking-only graph
        self.complete_graph = nx.DiGraph()  # Complete graph with walking
        
        # OSMnx graph for full Manila
        self.osmnx_graph = None  # Will be loaded for full Manila
        self.osmnx_nodes = {}  # Cache for OSMnx node coordinates
        self.shape_generator = None  # Will be initialized later
        
        # Route types cache
        self.route_types = {}  # Cache for route types
        
        # Configuration
        self.max_walking_distance = 0.8  # km - balanced for 8GB laptop
        self.max_transfer_distance = 0.6  # km - increased to 600m as requested
        self.walking_speed = 5.0  # km/h
        self.transfer_penalty = 3.0  # default penalty for transfers (km equivalent)
        self.same_mode_transfer_penalty = 0.5  # much lower penalty for same-mode transfers (e.g., jeep-jeep)
        
        # Mode weights (prefer transit over walking)
        self.mode_weights = {
            'LRT': 1.0,
            'Bus': 1.2,  # Slightly higher cost than LRT
            'Jeep': 1.5,  # Higher cost than bus
            'Walking': 3.0,  # Much higher cost to prefer transit
            'Transfer': 0.5  # Low cost for transfers
        }
        
        # Fare data
        self.fare_tables = {}

        # GeoJSON LineStrings for polylines
        self.street_lines = []  # List of LineStrings from fullcityofmanila.geojson
        self.lrt_lines = []     # List of LineStrings from lrtroutes.geojson
        
        # Mapbox API key no longer used – removed to avoid stale or invalid token exposure
        # OpenRouteService API key – environment variable takes precedence, but we default to the
        # user-supplied public key so the engine works immediately without additional setup.
        self.ors_api_key = os.getenv("ORS_API_KEY", "5b3ce3597851110001cf6248c12fb11ab4f84f55861d273a622d7aab")
        
        # Load GTFS data
        print("Loading GTFS data...")
        self._load_gtfs_data()
        
        # Load fare tables
        print("Loading fare tables...")
        self._load_fare_data()
        
        # Load and index GeoJSON LineStrings
        print("Loading and indexing GeoJSON LineStrings...")
        self._load_geojson_linestrings()
        
        # Load Manila OSMnx graph
        print("Loading Manila OSMnx graph...")
        self._load_manila_osmnx_graph()
        
        # Build routing graphs
        print("Building routing graphs...")
        self._build_graphs()
        
        # Load allowed transfers
        self.allowed_transfers = set()
        transfers_path = os.path.join(self.data_dir, 'transfers.txt')
        if os.path.exists(transfers_path):
            with open(transfers_path, 'r', encoding='utf-8') as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        self.allowed_transfers.add((parts[0], parts[1]))
        
        # -------------------------------------------------------------
        #  Tricycle terminals (optional last/first mile enhancement)
        # -------------------------------------------------------------
        # Load tricycle terminals (env var TRICYCLE_TERMINALS_GEOJSON overrides default)
        # Resolve terminal GeoJSON relative to the *package* directory so that
        # the engine always finds it irrespective of where the Flask app is
        # launched from (the default relative path fails when CWD != backend/routing).

        pkg_root = os.path.dirname(__file__)  # maiwayrouting/
        # Data folder lives one directory up: .../routing/data/
        trike_geojson = os.path.abspath(os.path.join(pkg_root, '..', self.data_dir, 'tricycle_terminals.geojson'))

        if not os.path.exists(trike_geojson):
            # Extra fallback – original relative lookup so existing setups keep working
            trike_geojson = os.path.join(self.data_dir, 'tricycle_terminals.geojson')

        self.trike_terminals = load_trike_terminals(trike_geojson)
        if self.trike_terminals:
            self.logger.info(f"Loaded {len(self.trike_terminals)} tricycle terminals")
        else:
            self.logger.info("No tricycle terminals loaded – convenient-mode trike feature disabled")
        
        print("MaiWay Routing Engine initialized successfully!")
    
    def _load_gtfs_data(self):
        """Load all GTFS data (stops, routes, trips, shapes, etc.)"""
        self.logger.info("Loading GTFS data for unified routing...")
        
        # Load stops
        try:
            stops_df = pd.read_csv(f"{self.data_dir}/stops.txt")
            for _, row in stops_df.iterrows():
                if pd.isnull(row['stop_id']) or pd.isnull(row['stop_lat']) or pd.isnull(row['stop_lon']):
                    self.logger.warning(f"Invalid stop row: {row}")
                    continue
                self.stops[row['stop_id']] = {
                    'name': row['stop_name'],
                    'lat': row['stop_lat'],
                    'lon': row['stop_lon'],
                    'zone_id': row.get('zone_id', '')
                }
        except Exception as e:
            self.logger.error(f"Failed to load stops.txt: {e}")
        
        # Load routes
        try:
            routes_df = pd.read_csv(f"{self.data_dir}/routes.txt")
            for _, row in routes_df.iterrows():
                self.routes[row['route_id']] = {
                    'short_name': row['route_short_name'],
                    'long_name': row['route_long_name'],
                    'route_type': row['route_type'],
                    'agency_id': row['agency_id']
                }
        except Exception as e:
            self.logger.error(f"Failed to load routes.txt: {e}")
        
        # Load trips
        try:
            trips_df = pd.read_csv(f"{self.data_dir}/trips.txt")
            for _, row in trips_df.iterrows():
                self.trips[row['trip_id']] = {
                    'route_id': row['route_id'],
                    'shape_id': row['shape_id'],
                    'direction_id': row.get('direction_id', 0)
                }
        except Exception as e:
            self.logger.error(f"Failed to load trips.txt: {e}")
        
        # Load stop times (IGNORE arrival/departure times as per prompt)
        try:
            stop_times_df = pd.read_csv(f"{self.data_dir}/stop_times.txt")
            for _, row in stop_times_df.iterrows():
                self.stop_times.append({
                    'trip_id': row['trip_id'],
                    'stop_sequence': row['stop_sequence'],
                    'stop_id': row['stop_id'],
                    'pickup_type': row.get('pickup_type', 0),
                    'drop_off_type': row.get('drop_off_type', 0)
                })
        except Exception as e:
            self.logger.error(f"Failed to load stop_times.txt: {e}")
        
        # Load transfers
        try:
            transfers_df = pd.read_csv(f"{self.data_dir}/transfers.txt")
            self.transfers = set()
            for _, row in transfers_df.iterrows():
                self.transfers.add((row['from_stop_id'], row['to_stop_id']))
            self.logger.info(f"Loaded {len(self.transfers)} transfers from transfers.txt")
        except Exception as e:
            self.logger.warning(f"Failed to load transfers.txt: {e}")
            self.transfers = set()
        
        # Load shapes
        self._load_shapes()
        
        self.logger.info(f"Loaded {len(self.stops)} stops, {len(self.routes)} routes, {len(self.trips)} trips, {len(self.stop_times)} stop times, {len(self.shapes)} shapes")
    
    def _load_manila_osmnx_graph(self):
        """Load cached Manila OSMnx graph for faster startup"""
        cache_file = os.path.join("cache", "manila_osmnx_graph.pkl")
        
        try:
            self.logger.info("Loading cached Manila OSMnx graph...")
            
            if not os.path.exists(cache_file):
                self.logger.warning(f"Cache file not found: {cache_file}")
                self.logger.warning("Run 'python scripts/download_manila_graph.py' to download the graph")
                self.osmnx_graph = None
                return
            
            # Load from cache
            with open(cache_file, 'rb') as f:
                self.osmnx_graph = pickle.load(f)
            
            # Extract node coordinates
            for node_id, node_data in self.osmnx_graph.nodes(data=True):
                if 'y' in node_data and 'x' in node_data:
                    self.osmnx_nodes[node_id] = (node_data['y'], node_data['x'])  # lat, lon
            
            self.logger.info(f"Loaded cached Manila OSMnx graph: {len(self.osmnx_graph.nodes)} nodes, {len(self.osmnx_graph.edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to load cached Manila OSMnx graph: {e}")
            self.osmnx_graph = None
            self.logger.warning("Falling back to basic routing without OSMnx graph")
    
    def _load_shapes(self):
        """Load GTFS shapes for snap-to-polyline functionality"""
        try:
            from .core_shape_generator import CoreShapeGenerator
            
            self.logger.info("Loading GTFS shapes for snap-to-polyline routing...")
            
            # Initialize shape generator
            self.shape_generator = CoreShapeGenerator(self.data_dir)
            self.shape_generator.set_stops_cache(self.stops)
            
            # Load original GTFS shapes
            shapes_df = pd.read_csv(f"{self.data_dir}/shapes.txt")
            self.logger.info(f"Processing {len(shapes_df)} GTFS shape points...")
            
            # Use vectorized operations for maximum speed
            shapes_df_sorted = shapes_df.sort_values(['shape_id', 'shape_pt_sequence'])
            
            # Group by shape_id and convert to list of tuples in one operation
            self.shapes = {}
            for shape_id, group in shapes_df_sorted.groupby('shape_id'):
                # Use numpy arrays for faster processing
                coords = list(zip(group['shape_pt_lat'].values, group['shape_pt_lon'].values))
                self.shapes[shape_id] = coords
            
            self.logger.info(f"Loaded {len(self.shapes)} OSMnx-patched shapes")
            
        except Exception as e:
            self.logger.warning(f"Could not load shapes for snap-to-polyline: {e}")
            self.shapes = {}
            self.shape_generator = None
    
    def _build_graphs(self):
        """Build all routing graphs"""
        self.logger.info("Building routing graphs...")
        # Build transit graph first (ONLY transit edges)
        self.logger.info("  Building transit graph...")
        self.transit_graph = build_transit_graph(self.stops, self.routes, self.trips, self.stop_times, self.mode_weights, self.logger)
        
        # Build walking graph separately (ONLY walking edges)
        self.logger.info("  Building walking graph...")
        self.walking_graph = nx.DiGraph()
        build_smart_walking_edges(self.stops, self.stop_times, self.trips, self.routes, self.mode_weights, self.transfer_penalty, self.logger, self.walking_graph, self.transfers)
        
        # Build complete graph by combining transit and walking graphs
        self.logger.info("  Building complete graph...")
        self.complete_graph = build_complete_graph(self.transit_graph, self.walking_graph, self.logger)
        
        # PERMANENT FIX: Ensure all stops are connected
        self.logger.info("  Applying permanent connectivity fix...")
        self._apply_connectivity_fix()
    
    # ---------------------------------------------------------------------
    #  NEW: Build CompleteRoute object from stop-id path with tiny-walk cleaner
    # ---------------------------------------------------------------------
    def _build_route_from_path(self, path: List[str]) -> CompleteRoute:
        """Convert a list of stop_ids from A* to a cleaned CompleteRoute"""
        segments: List[RouteSegment] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.complete_graph.get_edge_data(u, v) or {}
            mode_raw = str(edge_data.get('mode', 'walking'))
            mode = 'Walking' if mode_raw.lower() in ['walk', 'walking'] else mode_raw.capitalize()
            seg = RouteSegment(
                from_stop=u,
                to_stop=v,
                mode=mode,
                distance=float(edge_data.get('distance', 0.0)),
                trip_id=edge_data.get('trip_id'),
                route_id=edge_data.get('route_id'),
                reason=edge_data.get('type'),
                fare=float(edge_data.get('fare', 0.0))
            )
            segments.append(seg)

        # --- Clean tiny walking stubs (<100 m) between same route_id ---
        cleaned: List[RouteSegment] = []
        i = 0
        THRESH_KM = 1.0  # 1 km – merge micro-walks between stops on the same line (Jeep/Bus/LRT)
        while i < len(segments):
            seg = segments[i]
            if seg.mode == 'Walking' and seg.distance < THRESH_KM and cleaned and i + 1 < len(segments):
                prev_seg = cleaned[-1]
                next_seg = segments[i + 1]

                # Determine if prev and next belong to the *same* vehicle line.
                same_exact_route = prev_seg.route_id and prev_seg.route_id == next_seg.route_id

                def _base(r):
                    return (r or '').split('_')[0]  # drop _in / _out suffixes

                same_prefix_route = (
                    prev_seg.route_id and next_seg.route_id and
                    _base(prev_seg.route_id) == _base(next_seg.route_id)
                )

                same_mode = prev_seg.mode.lower() == next_seg.mode.lower() and prev_seg.mode.lower() in {'jeep', 'bus', 'lrt'}

                same_line = (same_exact_route or (same_mode and same_prefix_route))

                if same_line:
                    # Merge: stay on board, discard the tiny walk
                    prev_seg.to_stop = next_seg.to_stop
                    prev_seg.distance += seg.distance + next_seg.distance
                    prev_seg.fare += next_seg.fare
                    i += 2
                    continue  # skip walk + next_seg
            cleaned.append(seg)
            i += 1

        # --- NEW: Merge consecutive transit segments belonging to the same route (bus/jeep/LRT) ---
        merged: List[RouteSegment] = []
        for seg in cleaned:
            if (
                merged and
                seg.mode == merged[-1].mode and
                seg.route_id and seg.route_id == merged[-1].route_id and
                seg.mode.lower() != 'walking'
            ):
                # Extend previous segment
                merged[-1].to_stop = seg.to_stop
                merged[-1].distance += seg.distance
                merged[-1].fare += seg.fare  # fare will be recalculated later anyway
            else:
                merged.append(seg)
        cleaned = merged

        # Re-compute totals after merging
        total_distance = sum(s.distance for s in cleaned)
        walking_distance = sum(s.distance for s in cleaned if s.mode == 'Walking')
        transit_distance = total_distance - walking_distance
        transfers = 0
        for j in range(len(cleaned) - 1):
            a, b = cleaned[j], cleaned[j + 1]
            if a.mode != b.mode or a.route_id != b.route_id:
                transfers += 1
        modes_used: Set[str] = {s.mode for s in cleaned}
        return CompleteRoute(
            segments=cleaned,
            total_distance=total_distance,
            transit_distance=transit_distance,
            walking_distance=walking_distance,
            num_transfers=transfers,
            modes_used=modes_used,
        )

    # ---------------------------------------------------------------------
    #  NEW: OpenRouteService helper for high-quality walking polylines
    # ---------------------------------------------------------------------
    def _get_ors_walking_directions(self, from_coords: Tuple[float, float], to_coords: Tuple[float, float]):
        """Return a list of (lon, lat) tuples from ORS foot-walking directions. Returns None on failure."""
        try:
            import openrouteservice  # Lazy import – not required if user skips ORS
            client = openrouteservice.Client(key=self.ors_api_key, timeout=10)
            geojson = client.directions([from_coords, to_coords], profile='foot-walking', format='geojson')
            coordinates = geojson['features'][0]['geometry']['coordinates']
            # ORS returns [lon, lat]
            return [(lon, lat) for lon, lat in coordinates]
        except Exception as e:
            # Log at debug level to avoid spamming production logs
            self.logger.debug(f"ORS request failed: {e}")
            return None
    
    def find_nearest_stop(self, lat: float, lon: float) -> Tuple[str, float]:
        """Return the nearest stop_id and the haversine distance (km) to the given coordinates.

        This small utility replaced a previously implicit helper that was lost during
        recent refactors.  It scans the already-loaded ``self.stops`` dict – which is
        typically <15 k items for Metro Manila – and computes the great-circle
        distance to each stop.  For production workloads this could be optimised
        with a KD-Tree, but the linear scan remains <5 ms on modern machines and
        keeps external dependencies minimal.
        """
        if not self.stops:
            raise ValueError("GTFS stops data not loaded – cannot search for nearest stop.")

        nearest_stop_id: Optional[str] = None
        min_distance_km: float = float('inf')

        for stop_id, info in self.stops.items():
            d = haversine_distance(lat, lon, info['lat'], info['lon'])
            if d < min_distance_km:
                min_distance_km = d
                nearest_stop_id = stop_id

        # Safety fallback (should never happen unless self.stops is empty)
        if nearest_stop_id is None:
            # Return a dummy stop id and huge distance so caller can handle gracefully
            return "", float('inf')
        return nearest_stop_id, min_distance_km
    
    def _apply_connectivity_fix(self):
        """Apply permanent connectivity fix to ensure all stops are connected"""
        import networkx as nx
        
        # Check current connectivity
        components = list(nx.strongly_connected_components(self.complete_graph))
        largest_component = max(components, key=len)
        
        if len(largest_component) == len(self.stops):
            self.logger.info("  All stops already connected!")
            return
        
        self.logger.info(f"  Found {len(components)} components, largest has {len(largest_component)} stops")
        self.logger.info(f"  Connecting {len(self.stops) - len(largest_component)} isolated stops...")
        
        # Get all stops and isolated stops
        all_stops = list(self.stops.keys())
        isolated_stops = set(all_stops) - largest_component
        
        # Find a central stop in the largest component
        connected_stops_list = list(largest_component)
        center_stop = connected_stops_list[len(connected_stops_list) // 2]
        
        # Connect all isolated stops to the center stop
        walking_edges_added = 0
        for isolated_stop in isolated_stops:
            isolated_stop_info = self.stops[isolated_stop]
            center_stop_info = self.stops[center_stop]
            
            distance = haversine_distance(
                isolated_stop_info['lat'], isolated_stop_info['lon'],
                center_stop_info['lat'], center_stop_info['lon']
            )
            
            # Add walking edge with high penalty for long distances
            edge_data = {
                'trip_id': 'CRITICAL_WALKING',
                'route_id': 'CRITICAL_WALKING',
                'mode': 'Walking',
                'distance': distance,
                'weight': distance * 3.0 + 10.0,  # High penalty for long walks
                'stop_sequence_from': 0,
                'stop_sequence_to': 0,
                'type': 'critical_walking'
            }
            
            # Add bidirectional edges
            self.complete_graph.add_edge(isolated_stop, center_stop, **edge_data)
            self.complete_graph.add_edge(center_stop, isolated_stop, **edge_data)
            walking_edges_added += 2
        
        self.logger.info(f"  Added {walking_edges_added} critical walking edges")
        
        # Verify fix
        final_components = list(nx.strongly_connected_components(self.complete_graph))
        largest_final = max(final_components, key=len)
        
        if len(largest_final) == len(self.stops):
            self.logger.info(f"  SUCCESS: All {len(self.stops)} stops are now connected!")
        else:
            self.logger.warning(f"  Still {len(self.stops) - len(largest_final)} stops isolated")
    

    
    def get_route_type_name(self, route_type: int, agency_id: str = '', route_id: str = '', route_long_name: str = '') -> str:
        """Convert GTFS route_type to mode name, robustly detecting LRT routes."""
        # LRT routes: route_type == 0, or route_id/long_name contains 'lrt'
        if route_type == 0 or (route_id and 'lrt' in route_id.lower()) or (route_long_name and 'lrt' in route_long_name.lower()):
            return 'LRT'
        # Jeep routes (most common in this dataset)
        if route_type == 2:
            return 'jeepney'
        # Bus routes
        if route_type == 3:
            return 'bus'
        # Walking edges
        if route_id == 'WALKING' or route_id == 'Walking':
            return 'walking'
        # Default fallback
        return 'jeepney'
    
    def find_route_with_walking(self, origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, mode: str = 'fastest', fare_type: str = 'regular') -> Optional[Dict[str, Any]]:
        """Find route with walking to/from stops - IMPROVED WITH FIRST/LAST MILE and mode-specific cost function"""
        # ------------------------------------------------------------------
        # SHORT-WALK OPTIMISER: if origin & destination are within 0.3 km we
        # skip transit completely and return one walking segment (with OSMnx
        # snapped polyline when available).  The returned dict now mirrors the
        # structure produced by the normal pipeline so the front-end accepts it.
        # ------------------------------------------------------------------

        straight_km = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        # SHORT-WALK OPTIMISER active again: for origin–destination <300 m, return a walking-only route
        if straight_km < 0.300:  # 300 m
            # For very short trips we build a single walking segment. We try
            # to get a high-quality snapped polyline via OSMnx if available,
            # but since the helper is optional we swallow any AttributeError
            # and fall back to a simple straight-line polyline.
            snapped = None
            try:
                if hasattr(self, 'osmnx_graph') and self.osmnx_graph:
                    import networkx as nx
                    import osmnx as ox  # Local import
                    from_node = ox.nearest_nodes(self.osmnx_graph, origin_lon, origin_lat)
                    to_node = ox.nearest_nodes(self.osmnx_graph, dest_lon, dest_lat)
                    path = nx.shortest_path(self.osmnx_graph, from_node, to_node, weight='length')
                    snapped = [(self.osmnx_graph.nodes[n]['x'], self.osmnx_graph.nodes[n]['y']) for n in path]
            except Exception:
                snapped = None
            walk_segment = {
                'from_stop': {
                    'id': 'ORIGIN',
                    'name': 'ORIGIN',
                    'lat': origin_lat,
                    'lon': origin_lon
                },
                'to_stop': {
                    'id': 'DESTINATION',
                    'name': 'DESTINATION',
                    'lat': dest_lat,
                    'lon': dest_lon
                },
                'mode': 'Walking',
                'distance': straight_km,
                'fare': 0.0,
                'trip_id': 'WALKING',
                'route_id': 'WALKING',
                'reason': 'short_walk_only',
                'polyline': snapped or [[origin_lon, origin_lat], [dest_lon, dest_lat]],
                'polyline_source': 'osmnx' if snapped else 'straight_line'
            }

            est_time_min = int((straight_km / 4.0) * 60)  # WALKING_SPEED_KMH = 4

            base = {
                'segments': [walk_segment],
                'total_distance': round(straight_km, 3),
                'total_cost': 0.0,
                'estimated_time': est_time_min,
                'num_segments': 1,
                'fare_breakdown': {'walking': 0.0},
                'stops': [
                    {'name': 'ORIGIN', 'lat': origin_lat, 'lon': origin_lon},
                    {'name': 'DESTINATION', 'lat': dest_lat, 'lon': dest_lon}
                ]
            }

            base['summary'] = {
                key: {
                    'total_distance': base['total_distance'],
                    'total_cost': 0.0,
                    'estimated_time': est_time_min,
                    'num_segments': 1,
                    'fare_breakdown': {'walking': 0.0}
                } for key in ['fastest', 'cheapest', 'convenient']
            }

            return base
        
        # (Removed quick-exit for sub-250 m trips; always run normal routing logic)

        # ------------------------------------------------------------------
        #  NEW: Tricycle first-mile injection for "convenient" preference
        # ------------------------------------------------------------------
        first_mile_override = None  # (walk_seg, trike_seg, terminal)
        if mode == 'convenient' and getattr(self, 'trike_terminals', None):
            # Check if user starts near a trike terminal (<=400 m)
            term_res = nearest_terminal(origin_lat, origin_lon, self.trike_terminals, max_km=TRIKE_CATCHMENT_KM)
            if term_res is None:
                self.logger.info("[TRIKE-FIRST] No terminal within catchment radius %.0fm" % (TRIKE_CATCHMENT_KM*1000))  # uses updated constant
            else:
                terminal, walk_km_to_terminal = term_res
                board_stop_id, board_dist_km = self.find_nearest_stop(terminal['lat'], terminal['lon'])
                if TRIKE_MIN_DISTANCE_KM <= board_dist_km <= TRIKE_MAX_DISTANCE_KM:
                    # Build first-mile walking segment (origin -> terminal)
                    walk_seg = {
                        'from_stop': {
                            'id': 'ORIGIN',
                            'name': 'ORIGIN',
                            'lat': origin_lat,
                            'lon': origin_lon,
                        },
                        'to_stop': {
                            'id': terminal['id'],
                            'name': terminal.get('name', 'Trike Terminal'),
                            'lat': terminal['lat'],
                            'lon': terminal['lon'],
                        },
                        'mode': 'Walking',
                        'distance': walk_km_to_terminal,
                        'fare': 0.0,
                        'trip_id': 'WALKING',
                        'route_id': 'WALKING',
                        'reason': 'first_mile',
                        'polyline': [[origin_lon, origin_lat], [terminal['lon'], terminal['lat']]],
                        'polyline_source': 'straight_line',
                    }

                    # Build trike segment (terminal -> boarding stop)
                    trike_seg = build_trike_segment(
                        {
                            'id': terminal['id'],
                            'name': terminal.get('name', 'Trike Terminal'),
                            'lat': terminal['lat'],
                            'lon': terminal['lon'],
                        },
                        {
                            'id': board_stop_id,
                            'name': self.stops[board_stop_id]['name'],
                            'lat': self.stops[board_stop_id]['lat'],
                            'lon': self.stops[board_stop_id]['lon'],
                        },
                    )

                    # Polyline placeholder – straight line; will be enhanced later by polyline generator
                    trike_seg['polyline'] = [[terminal['lon'], terminal['lat']], [self.stops[board_stop_id]['lon'], self.stops[board_stop_id]['lat']]]
                    trike_seg['polyline_source'] = 'straight_line'

                    # Log for debugging
                    self.logger.info(
                        (
                            f"[TRIKE-FIRST] terminal={terminal['id']} walk_to_terminal={walk_km_to_terminal:.3f} km "
                            f"→ stop {board_stop_id} ride={board_dist_km:.3f} km – injecting first-mile tricycle"
                        )
                    )

                    first_mile_override = (walk_seg, trike_seg, board_stop_id)

                    # Override origin stop for A* search
                    origin_stop_override = board_stop_id
                else:
                    # Nearest stop is either too close (<0.5 km) or too far (>2 km)
                    limit = "< %.1f" % TRIKE_MIN_DISTANCE_KM if board_dist_km < TRIKE_MIN_DISTANCE_KM else "> %.1f" % TRIKE_MAX_DISTANCE_KM
                    self.logger.info(
                        (
                            f"[TRIKE-FIRST] terminal found but nearest stop {board_stop_id} distance {board_dist_km:.3f} km (outside useful range {limit} km)"
                        )
                    )
                    origin_stop_override = None
        else:
            origin_stop_override = None

        # Configuration for walking limits
        MAX_WALKING_TO_STOP = 1.3  # 1.3 km max walking to first/last stop
        WALKING_SPEED_KMH = 4.0    # Average walking speed
        
        # Find nearest stops to origin and destination (use override if set)
        if origin_stop_override:
            origin_stop_id = origin_stop_override
            origin_distance = 0.0  # already covered by trike segment
        else:
            origin_stop_id, origin_distance = self.find_nearest_stop(origin_lat, origin_lon)
        dest_stop_id, dest_distance = self.find_nearest_stop(dest_lat, dest_lon)
        
        self.logger.info(f"Nearest stops: origin={origin_stop_id} ({origin_distance:.3f} km), dest={dest_stop_id} ({dest_distance:.3f} km)")
        
        # Check if walking distances are reasonable
        if origin_distance > MAX_WALKING_TO_STOP:
            self.logger.warning(f"Origin too far from nearest stop: {origin_distance:.3f} km > {MAX_WALKING_TO_STOP} km")
            return None
        
        if dest_distance > MAX_WALKING_TO_STOP:
            self.logger.warning(f"Destination too far from nearest stop: {dest_distance:.3f} km > {MAX_WALKING_TO_STOP} km")
            return None
        
        # Find route between stops using the selected mode
        route = self.find_route(origin_stop_id, dest_stop_id, mode=mode, prefer_transit=True, max_transfers=3)
        
        if not route:
            self.logger.warning(f"No route found between stops {origin_stop_id} and {dest_stop_id}")
            if origin_distance + dest_distance < 2.0:  # If total walking distance is reasonable
                self.logger.info("Creating walking-only route as fallback")
                segments = []
                
                # Add first mile walking if origin is not at a stop
                if origin_distance > 0.001:
                    first_mile_walking = {
                        'from_stop': {
                            'id': 'ORIGIN',
                            'name': 'ORIGIN',
                            'lat': origin_lat,
                            'lon': origin_lon
                        },
                        'to_stop': {
                            'id': origin_stop_id,
                            'name': self.stops[origin_stop_id]['name'],
                            'lat': self.stops[origin_stop_id]['lat'],
                            'lon': self.stops[origin_stop_id]['lon']
                        },
                        'mode': 'Walking',
                        'distance': origin_distance,
                        'fare': 0.0,
                        'trip_id': 'WALKING',
                        'route_id': 'WALKING',
                        'reason': 'first_mile',
                        'polyline': [[origin_lon, origin_lat], [self.stops[origin_stop_id]['lon'], self.stops[origin_stop_id]['lat']]],
                        'polyline_source': 'straight_line'
                    }
                    segments.append(first_mile_walking)
                
                # Add last mile walking if destination is not at a stop
                if dest_distance > 0.001:
                    last_mile_walking = {
                        'from_stop': {
                            'id': dest_stop_id,
                            'name': self.stops[dest_stop_id]['name'],
                            'lat': self.stops[dest_stop_id]['lat'],
                            'lon': self.stops[dest_stop_id]['lon']
                        },
                        'to_stop': {
                            'id': 'DESTINATION',
                            'name': 'DESTINATION',
                            'lat': dest_lat,
                            'lon': dest_lon
                        },
                        'mode': 'Walking',
                        'distance': dest_distance,
                        'fare': 0.0,
                        'trip_id': 'WALKING',
                        'route_id': 'WALKING',
                        'reason': 'last_mile',
                        'polyline': [[self.stops[dest_stop_id]['lon'], self.stops[dest_stop_id]['lat']], [dest_lon, dest_lat]],
                        'polyline_source': 'straight_line'
                    }
                    segments.append(last_mile_walking)
                
                # Calculate totals
                total_distance = sum(seg.get('distance', 0) for seg in segments)
                total_fare = sum(seg.get('fare', 0) for seg in segments)
                total_time = int(total_distance / WALKING_SPEED_KMH * 60)  # Convert to minutes
                
                return {
                    'segments': segments,
                    'total_distance': total_distance,
                    'total_cost': total_fare,
                    'estimated_time': total_time,
                    'num_segments': len(segments),
                    'fare_breakdown': self._calculate_fare_breakdown(segments),
                    'summary': {
                        'fastest': {
                            'total_distance': total_distance,
                            'total_cost': total_fare,
                            'estimated_time': total_time,
                            'num_segments': len(segments),
                            'fare_breakdown': self._calculate_fare_breakdown(segments)
                        },
                        'cheapest': {
                            'total_distance': total_distance,
                            'total_cost': total_fare,
                            'estimated_time': total_time,
                            'num_segments': len(segments),
                            'fare_breakdown': self._calculate_fare_breakdown(segments)
                        },
                        'convenient': {
                            'total_distance': total_distance,
                            'total_cost': total_fare,
                            'estimated_time': total_time,
                            'num_segments': len(segments),
                            'fare_breakdown': self._calculate_fare_breakdown(segments)
                        }
                    }
                }
            return None
        
        # --------------------------------------------
        #  Inject trike + walk segments if we built one
        # --------------------------------------------
        segments = list(route.segments)
        if first_mile_override:
            walk_seg, trike_seg, _ = first_mile_override
            segments = [walk_seg, trike_seg] + segments

        # Add first mile walking if origin is not at a stop
        first_mile_walking = None
        if origin_distance > 0.001 and not first_mile_override:  # Skip default if trike handled it
            first_mile_walking = {
                'from_stop': {
                    'id': 'ORIGIN',
                    'name': 'ORIGIN',
                    'lat': origin_lat,
                    'lon': origin_lon
                },
                'to_stop': {
                    'id': origin_stop_id,
                    'name': self.stops[origin_stop_id]['name'],
                    'lat': self.stops[origin_stop_id]['lat'],
                    'lon': self.stops[origin_stop_id]['lon']
                },
                'mode': 'Walking',
                'distance': origin_distance,
                'fare': 0.0,
                'trip_id': 'WALKING',
                'route_id': 'WALKING',
                'reason': 'first_mile',
                'polyline': [[origin_lon, origin_lat], [self.stops[origin_stop_id]['lon'], self.stops[origin_stop_id]['lat']]],
                'polyline_source': 'straight_line',
                'from_lon': origin_lon,
                'from_lat': origin_lat,
                'to_lon': self.stops[origin_stop_id]['lon'],
                'to_lat': self.stops[origin_stop_id]['lat'],
                'origin_lon': origin_lon,
                'origin_lat': origin_lat
            }
        
        # Add last mile walking if destination is not at a stop
        last_mile_walking = None
        if dest_distance > 0.001:  # If destination is not exactly at a stop
            last_mile_walking = {
                'from_stop': {
                    'id': dest_stop_id,
                    'name': self.stops[dest_stop_id]['name'],
                    'lat': self.stops[dest_stop_id]['lat'],
                    'lon': self.stops[dest_stop_id]['lon']
                },
                'to_stop': {
                    'id': 'DESTINATION',
                    'name': 'DESTINATION',
                    'lat': dest_lat,
                    'lon': dest_lon
                },
                'mode': 'Walking',
                'distance': dest_distance,
                'fare': 0.0,
                'trip_id': 'WALKING',
                'route_id': 'WALKING',
                'reason': 'last_mile',
                'polyline': [[self.stops[dest_stop_id]['lon'], self.stops[dest_stop_id]['lat']], [dest_lon, dest_lat]],
                'polyline_source': 'straight_line',
                'from_lon': self.stops[dest_stop_id]['lon'],
                'from_lat': self.stops[dest_stop_id]['lat'],
                'to_lon': dest_lon,
                'to_lat': dest_lat,
                'dest_lon': dest_lon,
                'dest_lat': dest_lat
            }
        
        # Build complete route with walking (or tricycle) segments
        segments = []

        # Priority: if a tricycle first-mile override was generated, inject both the
        # initial walk-to-terminal and the trike ride *before* any transit
        # segments.  Otherwise fall back to a standard first-mile walking
        # segment when needed.
        if first_mile_override:
            walk_seg, trike_seg, _ = first_mile_override
            segments.extend([walk_seg, trike_seg])
        elif first_mile_walking:
            segments.append(first_mile_walking)
        
        # -------------------------------------------------------------
        #  NEW: Optional last-mile tricycle injection (convenient only)
        #  – applies ONLY if we did NOT already use a first-mile trike
        # -------------------------------------------------------------
        last_mile_override = None  # (trike_seg, walk_seg)
        # Last-mile tricycle temporarily disabled – condition short-circuited
        if False and (first_mile_override is None  # never allow two trike legs
                and mode == 'convenient'
                and getattr(self, 'trike_terminals', None)):

            term_res = nearest_terminal(dest_lat, dest_lon, self.trike_terminals, max_km=TRIKE_CATCHMENT_KM)
            if term_res is not None:
                terminal, walk_km_from_terminal = term_res

                km_stop_to_terminal = haversine_distance(
                    terminal['lat'], terminal['lon'],
                    self.stops[dest_stop_id]['lat'], self.stops[dest_stop_id]['lon']
                )

                if km_stop_to_terminal <= TRIKE_MAX_DISTANCE_KM:
                    # Build trike segment (from last GTFS stop to terminal)
                    trike_seg = build_trike_segment(
                        {
                            'id': dest_stop_id,
                            'name': self.stops[dest_stop_id]['name'],
                            'lat': self.stops[dest_stop_id]['lat'],
                            'lon': self.stops[dest_stop_id]['lon'],
                        },
                        {
                            'id': terminal['id'],
                            'name': terminal.get('name', 'Trike Terminal'),
                            'lat': terminal['lat'],
                            'lon': terminal['lon'],
                        },
                    )

                    trike_seg['polyline'] = [[self.stops[dest_stop_id]['lon'], self.stops[dest_stop_id]['lat']],
                                             [terminal['lon'], terminal['lat']]]
                    trike_seg['polyline_source'] = 'straight_line'

                    # Build short walk from terminal to final destination
                    walk_seg_dest = {
                        'from_stop': {
                            'id': terminal['id'],
                            'name': terminal.get('name', 'Trike Terminal'),
                            'lat': terminal['lat'],
                            'lon': terminal['lon'],
                        },
                        'to_stop': {
                            'id': 'DESTINATION',
                            'name': 'DESTINATION',
                            'lat': dest_lat,
                            'lon': dest_lon,
                        },
                        'mode': 'Walking',
                        'distance': walk_km_from_terminal,
                        'fare': 0.0,
                        'trip_id': 'WALKING',
                        'route_id': 'WALKING',
                        'reason': 'last_mile',
                        'polyline': [[terminal['lon'], terminal['lat']], [dest_lon, dest_lat]],
                        'polyline_source': 'straight_line',
                    }

                    # Log for debugging
                    self.logger.info(
                        (
                            f"[TRIKE-LAST] stop→terminal={km_stop_to_terminal:.3f} km walk_terminal→dest={walk_km_from_terminal:.3f} km – injecting last-mile tricycle"
                        )
                    )

                    last_mile_override = (trike_seg, walk_seg_dest)

                    # Override dest_distance so default last-mile walking is skipped
                    dest_distance = 0.0

        # Add transit segments
        for segment in route.segments:
            # Convert RouteSegment to dict format
            # Build rich stop objects (id, name, lat, lon) for nicer API output
            def _stop_meta(stop_id: str):
                info = self.stops.get(stop_id, {})
                return {
                    'id': stop_id,
                    'name': info.get('name', stop_id),
                    'lat': info.get('lat', 0.0),
                    'lon': info.get('lon', 0.0)
                }

            segment_dict = {
                'from_stop': _stop_meta(segment.from_stop),
                'to_stop': _stop_meta(segment.to_stop),
                'mode': segment.mode,
                'distance': segment.distance,
                'fare': self.calculate_real_fare(segment.mode, segment.from_stop, segment.to_stop, segment.distance, fare_type),
                'trip_id': segment.trip_id,
                'route_id': segment.route_id,
                'reason': segment.reason or 'transit',
                'polyline': [],  # Will be filled by polyline generation
                'polyline_source': 'GTFS'
            }
            segments.append(segment_dict)
        
        # Append last-mile override if we built one, otherwise default walk
        if last_mile_override:
            trike_seg, walk_dest_seg = last_mile_override
            segments.extend([trike_seg, walk_dest_seg])
        elif last_mile_walking:
            segments.append(last_mile_walking)
 
        # --- NEW: Coalesce consecutive walking segments (handles start/end duplicates) ---
        merged_segments = []
        for seg in segments:
            if merged_segments and seg.get('mode') == 'Walking' and merged_segments[-1].get('mode') == 'Walking':
                # Extend previous walking segment
                merged_segments[-1]['to_stop'] = seg.get('to_stop')
                merged_segments[-1]['distance'] += seg.get('distance', 0)
                merged_segments[-1]['fare'] += seg.get('fare', 0)
                # Reset polyline so a fresh one is generated later
                merged_segments[-1]['polyline'] = []
                merged_segments[-1]['polyline_source'] = 'merged'
                continue
            merged_segments.append(seg)
        segments = merged_segments
        
        # Generate polylines for all segments
        segments = self._generate_polylines_for_grouped_segments(segments)
        
        # Calculate totals
        total_distance = round(sum(seg.get('distance', 0) for seg in segments), 2)
        total_fare = round(sum(seg.get('fare', 0) for seg in segments), 2)
        
        # Estimate travel time in minutes using dedicated helper (correct hours→minutes conversion)
        total_time = self._estimate_travel_time(segments)
        
        # Build response
        response = {
            'segments': segments,
            'total_distance': total_distance,
            'total_cost': total_fare,
            'estimated_time': total_time,
            'num_segments': len(segments),
            'fare_breakdown': self._calculate_fare_breakdown(segments),
            'summary': {
                'fastest': {
                    'total_distance': total_distance,
                    'total_cost': total_fare,
                    'estimated_time': total_time,
                    'num_segments': len(segments),
                    'fare_breakdown': self._calculate_fare_breakdown(segments)
                },
                'cheapest': {
                    'total_distance': total_distance,
                    'total_cost': total_fare,
                    'estimated_time': total_time,
                    'num_segments': len(segments),
                    'fare_breakdown': self._calculate_fare_breakdown(segments)
                },
                'convenient': {
                    'total_distance': total_distance,
                    'total_cost': total_fare,
                    'estimated_time': total_time,
                    'num_segments': len(segments),
                    'fare_breakdown': self._calculate_fare_breakdown(segments)
                }
            }
        }
        
        return response
    
    def _generate_instruction(self, segment: Dict[str, Any]) -> str:
        """Generate human-readable instruction for a segment"""
        mode = segment.get('mode', 'Walking')
        
        # Handle both string and dict for from_stop/to_stop
        from_stop_raw = segment.get('from_stop', 'Unknown')
        to_stop_raw = segment.get('to_stop', 'Unknown')
        
        if isinstance(from_stop_raw, dict):
            from_stop = from_stop_raw.get('name', from_stop_raw.get('id', 'Unknown'))
        else:
            from_stop = str(from_stop_raw)
            
        if isinstance(to_stop_raw, dict):
            to_stop = to_stop_raw.get('name', to_stop_raw.get('id', 'Unknown'))
        else:
            to_stop = str(to_stop_raw)
        
        if mode == 'Walking':
            if segment.get('reason') == 'first_mile':
                return f"Walk from origin to {to_stop}"
            elif segment.get('reason') == 'last_mile':
                return f"Walk from {from_stop} to destination"
            else:
                return f"Walk from {from_stop} to {to_stop}"
        elif mode in ['LRT', 'Bus', 'Jeep']:
            route_id = segment.get('route_id', '')
            return f"Take {mode} {route_id} from {from_stop} to {to_stop}"
        else:
            return f"Take {mode} from {from_stop} to {to_stop}"
    
    def _estimate_travel_time(self, segments: List[Dict[str, Any]]) -> int:
        """Estimate total travel time in minutes"""
        total_time = 0
        
        for segment in segments:
            mode = segment.get('mode', 'Walking')
            distance = segment.get('distance', 0.0)
            
            if mode == 'Walking':
                # Walking speed: 5 km/h = 0.083 km/min
                time = distance / 0.083
            elif mode == 'LRT':
                # LRT speed: ~30 km/h = 0.5 km/min
                time = distance / 0.5
            elif mode in ['Bus', 'Jeep']:
                # Bus/Jeep speed: ~20 km/h = 0.33 km/min
                time = distance / 0.33
            else:
                # Default speed: 15 km/h = 0.25 km/min
                time = distance / 0.25
            
            total_time += time
        
        return int(total_time)
    
    def _calculate_fare_breakdown(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate fare breakdown by mode"""
        breakdown = {}
        
        for segment in segments:
            mode = segment.get('mode', 'Walking')
            fare = segment.get('fare', 0.0)
            
            if mode not in breakdown:
                breakdown[mode] = 0.0
            breakdown[mode] += fare
        # Round to 2 decimals for cleaner UI
        return {k: round(v, 2) for k, v in breakdown.items()}

    def _generate_polylines_for_grouped_segments(self, grouped_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate polylines for grouped segments with *mode-aware* preference order:

        • Walking   →  ORS directions → OSMnx snap → straight line
        • Transit   →  GTFS shape slice → LRT-specific geojson (if rail) →
                         OSMnx snap → fullcityofmanila.geojson → route-specific
                         street geojsons → straight line

        For every candidate polyline we run quick sanity checks (mainly length
        ratio to straight-line distance) and discard obviously wrong results
        before falling back to the next source.
        """
        import networkx as nx
        import pickle
        from shapely.geometry import LineString, Point
        def get_osmnx_snapped_polyline(from_coords, to_coords):
            if not self.osmnx_graph:
                return None
            try:
                import osmnx as ox  # local import to avoid global SSL issues
                from_node = ox.nearest_nodes(self.osmnx_graph, from_coords[0], from_coords[1])
                to_node = ox.nearest_nodes(self.osmnx_graph, to_coords[0], to_coords[1])
                path = nx.shortest_path(self.osmnx_graph, from_node, to_node, weight='length')
                polyline = [(self.osmnx_graph.nodes[n]['x'], self.osmnx_graph.nodes[n]['y']) for n in path]
                return polyline
            except Exception as e:
                self.logger.warning(f"OSMnx polyline failed: {e}")
                return None
        cache_file = os.path.join('cache', 'snapped_polylines_cache.pkl')
        try:
            with open(cache_file, 'rb') as f:
                snapped_cache = pickle.load(f)
        except Exception:
            snapped_cache = {}
        def cache_key(from_coords, to_coords, mode):
            return f"{mode}:{from_coords[0]:.6f},{from_coords[1]:.6f}->{to_coords[0]:.6f},{to_coords[1]:.6f}"
        enhanced_segments = []
        for group in grouped_segments:
            enhanced_segment = group.copy()
            # Ensure snapped_polyline is defined for all execution paths
            snapped_polyline: list = None
            from_stop_raw = group.get('from_stop', {})
            to_stop_raw = group.get('to_stop', {})
            from_coords = None
            to_coords = None
            if from_stop_raw == 'ORIGIN':
                from_coords = (group.get('start_lon', group.get('origin_lon', 0)), group.get('start_lat', group.get('origin_lat', 0)))
            elif isinstance(from_stop_raw, str) and from_stop_raw in self.stops:
                from_coords = (self.stops[from_stop_raw].get('lon', 0), self.stops[from_stop_raw].get('lat', 0))
            elif isinstance(group.get('from_stop'), dict):
                from_coords = (group['from_stop'].get('lon', 0), group['from_stop'].get('lat', 0))
            if to_stop_raw == 'DESTINATION':
                to_coords = (group.get('end_lon', group.get('dest_lon', 0)), group.get('end_lat', group.get('dest_lat', 0)))
            elif isinstance(to_stop_raw, str) and to_stop_raw in self.stops:
                to_coords = (self.stops[to_stop_raw].get('lon', 0), self.stops[to_stop_raw].get('lat', 0))
            elif isinstance(group.get('to_stop'), dict):
                to_coords = (group['to_stop'].get('lon', 0), group['to_stop'].get('lat', 0))
            if not from_coords or not to_coords or from_coords == (0,0) or to_coords == (0,0):
                self.logger.error(f"[POLYLINE] Invalid coordinates for segment: from_stop={from_stop_raw}, to_stop={to_stop_raw}, from_coords={from_coords}, to_coords={to_coords}")
            if from_coords and to_coords:
                try:
                    mode = group.get('mode', '').lower()
                    # First preference: ORS walking directions for walking legs
                    if mode in ['walk', 'walking']:
                        ors_poly = self._get_ors_walking_directions(from_coords, to_coords)
                        if ors_poly and len(ors_poly) > 1:
                            enhanced_segment['polyline'] = ors_poly
                            enhanced_segment['polyline_source'] = 'openrouteservice'
                            snapped_polyline = None  # skip OSMnx if ORS succeeded
                    trip_id = group.get('trip_id', '')

                    # Helper to extract proper stop_id string (handles dict vs str)
                    def _stop_id(raw):
                        if isinstance(raw, str):
                            return raw
                        if isinstance(raw, dict):
                            return str(raw.get('id') or raw.get('stop_id') or raw.get('name') or raw)
                        return str(raw)

                    from_id = _stop_id(from_stop_raw)
                    to_id   = _stop_id(to_stop_raw)

                    key = cache_key(from_coords, to_coords, mode)

                    # --------------------------------------------------
                    # PREFERENCE ORDER (per segment type)
                    #   WALKING: ORS → OSMnx-snap → straight
                    #   TRANSIT (bus/jeep/etc): GTFS shape → OSMnx-snap → street/lrt geojson → straight
                    # --------------------------------------------------

                    # 1) For walking – already tried ORS above; if it succeeded we skip remainder
                    if mode in ['walk', 'walking'] and enhanced_segment.get('polyline'):
                        pass  # Already have ORS polyline
                    else:
                        # 2) Transit: try GTFS shape FIRST (before OSMnx) except LRT handled later
                        if not enhanced_segment.get('polyline') and mode not in ['walk', 'walking'] and trip_id and trip_id.lower() not in ['walking', 'lrt_shape']:
                            shape_points = self.shape_generator.slice_shape_between_stops(
                                trip_id, from_id, to_id
                            )
                            if shape_points and len(shape_points) > 1:

                                # -------- Sanity-check GTFS slice length --------
                                try:
                                    from maiwayrouting.utils.geo_utils import haversine_distance

                                    def _poly_len(coords):
                                        return sum(
                                            haversine_distance(coords[i][1], coords[i][0], coords[i+1][1], coords[i+1][0])
                                            for i in range(len(coords)-1)
                                        )

                                    straight_dist = haversine_distance(from_coords[1], from_coords[0], to_coords[1], to_coords[0])
                                    path_len = _poly_len(shape_points)
                                    ratio = path_len / straight_dist if straight_dist else 1.0

                                    # --- Collinearity/curvature check: reject if all points are close to straight line ---
                                    def _point_line_distance(lat, lon, lat1, lon1, lat2, lon2):
                                        # Returns distance in meters from (lat,lon) to line (lat1,lon1)-(lat2,lon2)
                                        # Uses equirectangular projection for small distances
                                        import math
                                        R = 6371000  # Earth radius in meters
                                        x0, y0 = math.radians(lon), math.radians(lat)
                                        x1, y1 = math.radians(lon1), math.radians(lat1)
                                        x2, y2 = math.radians(lon2), math.radians(lat2)
                                        dx = x2 - x1
                                        dy = y2 - y1
                                        if dx == 0 and dy == 0:
                                            # Start and end are the same point
                                            return math.hypot(x0 - x1, y0 - y1) * R
                                        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
                                        proj_x = x1 + t * dx
                                        proj_y = y1 + t * dy
                                        return math.hypot(x0 - proj_x, y0 - proj_y) * R

                                    collinear_threshold_m = 15  # meters (stricter)
                                    collinear = False
                                    max_dist = 0
                                    total_angle = 0
                                    if len(shape_points) > 2:
                                        lat1, lon1 = from_coords[1], from_coords[0]
                                        lat2, lon2 = to_coords[1], to_coords[0]
                                        for pt in shape_points:
                                            d = _point_line_distance(pt[1], pt[0], lat1, lon1, lat2, lon2)
                                            if d > max_dist:
                                                max_dist = d
                                        if max_dist < collinear_threshold_m:
                                            collinear = True
                                        # Total angle check
                                        import math
                                        def _angle(a, b, c):
                                            ab = (b[0]-a[0], b[1]-a[1])
                                            bc = (c[0]-b[0], c[1]-b[1])
                                            norm_ab = math.hypot(*ab)
                                            norm_bc = math.hypot(*bc)
                                            if norm_ab == 0 or norm_bc == 0:
                                                return 0
                                            dot = ab[0]*bc[0] + ab[1]*bc[1]
                                            cos_angle = dot / (norm_ab * norm_bc)
                                            angle = math.acos(max(-1, min(1, cos_angle)))
                                            return abs(math.degrees(angle))
                                        total_angle = 0
                                        for i in range(1, len(shape_points)-1):
                                            a = shape_points[i-1]
                                            b = shape_points[i]
                                            c = shape_points[i+1]
                                            total_angle += _angle(a, b, c)
                                    angle_too_low = total_angle < 15

                                    suspect_few_points = straight_dist > 0.3 and len(shape_points) <= 4
                                    near_straight = straight_dist > 0.3 and ratio <= 1.08

                                    if straight_dist > 0 and (
                                        path_len > straight_dist * 2 or
                                        path_len - straight_dist > 10 or
                                        ratio < 0.7 or
                                        near_straight or
                                        suspect_few_points or
                                        collinear or
                                        angle_too_low
                                    ):
                                        self.logger.warning(
                                            (
                                                f"[POLYLINE] GTFS slice {trip_id} {from_id}->{to_id} failed sanity check: "
                                                f"len={path_len:.2f} km, straight={straight_dist:.2f} km, "
                                                f"ratio={ratio:.2f}, points={len(shape_points)}, collinear={collinear}, max_dist={max_dist:.1f}m, total_angle={total_angle:.1f}deg; discarding."
                                            )
                                        )
                                    else:
                                        enhanced_segment['polyline'] = shape_points
                                        enhanced_segment['polyline_source'] = 'gtfs_shape'
                                except Exception as e:
                                    self.logger.warning(f"[POLYLINE] Length sanity check for GTFS slice failed: {e}")
                                    # still accept the shape tentatively; may be replaced by later sanity checks
                                    enhanced_segment['polyline'] = shape_points
                                    enhanced_segment['polyline_source'] = 'gtfs_shape'

                    # 3) OSMnx-snap (roads) – use only if still nothing and mode != 'lrt'
                    if (not enhanced_segment.get('polyline') or len(enhanced_segment['polyline']) < 2) and mode != 'lrt':
                        snapped_polyline = snapped_cache.get(key)
                        if snapped_polyline is None:
                            snapped_polyline = get_osmnx_snapped_polyline(from_coords, to_coords)
                            if snapped_polyline and len(snapped_polyline) > 1:
                                snapped_cache[key] = snapped_polyline

                    # If we obtained a valid snapped polyline, prefer it
                    if snapped_polyline and len(snapped_polyline) > 1:
                        enhanced_segment['polyline'] = snapped_polyline
                        enhanced_segment['polyline_source'] = 'osmnx_snapped'

                        # --- Sanity check: if snapped path is wildly longer than straight line,
                        # it likely contains loops.  If so, fall back to straight-line polyline.
                        try:
                            from maiwayrouting.utils.geo_utils import haversine_distance as _h
                            def _poly_len(coords):
                                return sum(
                                    _h(coords[i][1], coords[i][0], coords[i+1][1], coords[i+1][0])
                                    for i in range(len(coords)-1)
                                )
                            straight_dist = haversine_distance(from_coords[1], from_coords[0], to_coords[1], to_coords[0])
                            path_len = _poly_len(snapped_polyline)
                            if straight_dist > 0 and path_len > straight_dist * 2:
                                self.logger.warning(
                                    f"[POLYLINE] Snapped path {from_stop_raw}->{to_stop_raw} is {path_len:.2f} km (>2× straight {straight_dist:.2f} km); "
                                    "falling back to straight line to avoid loops."
                                )
                                enhanced_segment['polyline'] = [from_coords, to_coords]
                                enhanced_segment['polyline_source'] = 'straight_line_sanity'
                        except Exception as e:
                            self.logger.warning(f"[POLYLINE] Length sanity check failed: {e}")
                    else:
                        # --------------------------------------------------
                        # (2) Dedicated LRT geojson (rail track) – only for rail
                        # --------------------------------------------------
                        if (not enhanced_segment.get('polyline') or len(enhanced_segment['polyline']) < 2) and mode == 'lrt' and hasattr(self, 'lrt_lines'):
                            best_line = None
                            min_dist = float('inf')
                            from_pt = Point(from_coords)
                            to_pt = Point(to_coords)
                            for line in self.lrt_lines:
                                geom = line['geometry']
                                dist = from_pt.distance(geom) + to_pt.distance(geom)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_line = geom
                            if best_line and isinstance(best_line, LineString):
                                # Slice the LineString between the two nearest vertices to get the correct segment only.
                                coords = list(best_line.coords)
                                # Helper to find nearest vertex index
                                def _nearest_idx(pt, coord_list):
                                    min_d, idx = float('inf'), 0
                                    for k, (x, y) in enumerate(coord_list):
                                        d = (x - pt[0])**2 + (y - pt[1])**2
                                        if d < min_d:
                                            min_d, idx = d, k
                                    return idx
                                idx_from = _nearest_idx(from_coords, coords)
                                idx_to   = _nearest_idx(to_coords, coords)
                                if idx_from > idx_to:
                                    idx_from, idx_to = idx_to, idx_from
                                sliced = coords[idx_from:idx_to + 1] if idx_to > idx_from else coords[idx_from:idx_from + 1]
                                # Ensure at least two points
                                if len(sliced) < 2:
                                    sliced = [from_coords, to_coords]
                                enhanced_segment['polyline'] = sliced
                                enhanced_segment['polyline_source'] = 'lrt_geojson'

                        # --------------------------------------------------
                        #  (4) fullcityofmanila.geojson (streets)
                        # --------------------------------------------------
                        if (not enhanced_segment.get('polyline') or len(enhanced_segment['polyline']) < 2) and hasattr(self, 'street_lines'):
                            best_line = None
                            min_dist = float('inf')
                            from_pt = Point(from_coords)
                            to_pt = Point(to_coords)
                            for line in getattr(self, 'street_lines', []):
                                geom = line['geometry']
                                dist = from_pt.distance(geom) + to_pt.distance(geom)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_line = geom
                            if best_line and isinstance(best_line, LineString):
                                enhanced_segment['polyline'] = list(best_line.coords)
                                enhanced_segment['polyline_source'] = 'fullcityofmanila_geojson'
                                self.logger.warning(f"Used fullcityofmanila.geojson fallback for {from_stop_raw}->{to_stop_raw}")
                        # (5) Try each geojson in data/routes-geojson
                        if (not enhanced_segment.get('polyline') or len(enhanced_segment['polyline']) < 2) and hasattr(self, 'route_lines'):
                            best_line = None
                            min_dist = float('inf')
                            for fname, lines in self.route_lines.items():
                                for line in lines:
                                    geom = line['geometry']
                                    dist = from_pt.distance(geom) + to_pt.distance(geom)
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_line = geom
                            if best_line and isinstance(best_line, LineString):
                                enhanced_segment['polyline'] = list(best_line.coords)
                                enhanced_segment['polyline_source'] = 'routes_geojson_fallback'
                                self.logger.warning(f"Used routes-geojson fallback ({fname}) for {from_stop_raw}->{to_stop_raw}")
                        # (6) Straight line as last resort
                        if not enhanced_segment.get('polyline') or len(enhanced_segment['polyline']) < 2:
                            enhanced_segment['polyline'] = [from_coords, to_coords]
                            enhanced_segment['polyline_source'] = 'straight_line'
                            self.logger.warning(f"Used straight line fallback for {from_stop_raw}->{to_stop_raw}")
                except Exception as e:
                    self.logger.warning(f"Error generating polyline: {e}")
                    enhanced_segment['polyline'] = [from_coords, to_coords]
                    enhanced_segment['polyline_source'] = 'straight_line'
            else:
                enhanced_segment['polyline'] = []
                enhanced_segment['polyline_source'] = 'invalid_coords'
                self.logger.warning(f"Invalid coordinates for grouped segment {from_stop_raw}->{to_stop_raw}")
                if from_coords is None:
                    self.logger.debug(f"  from_stop_raw: {from_stop_raw}")
                    self.logger.debug(f"  from_coords: {from_coords}")
                if to_coords is None:
                    self.logger.debug(f"  to_stop_raw: {to_stop_raw}")
                    self.logger.debug(f"  to_coords: {to_coords}")
            if not enhanced_segment.get('polyline') or len(enhanced_segment['polyline']) < 2:
                self.logger.error(f"Segment {from_stop_raw}->{to_stop_raw} has empty polyline, forcing fallback.")
                if from_coords and to_coords:
                    enhanced_segment['polyline'] = [from_coords, to_coords]
                else:
                    enhanced_segment['polyline'] = [(0,0), (0,0)]
                enhanced_segment['polyline_source'] = 'forced_fallback'
            enhanced_segments.append(enhanced_segment)
        # Save updated cache
        try:
            # Ensure cache directory exists
            os.makedirs('cache', exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(snapped_cache, f)
        except Exception as e:
            self.logger.warning(f"Failed to save snapped polyline cache: {e}")
        return enhanced_segments

    def _load_fare_data(self):
        """Load fare data from CSV files"""
        self.logger.info("Loading fare data...")
        try:
            # Load distance-based fare tables
            bus_fares = pd.read_csv(os.path.join(self.data_dir, "fares", "bus.csv"))
            jeep_fares = pd.read_csv(os.path.join(self.data_dir, "fares", "jeep.csv"))
            # LRT has two matrices: SJ (single-journey regular) & SV (stored-value discounted)
            lrt_sj = pd.read_csv(os.path.join(self.data_dir, "fares", "lrt1_sj.csv"), index_col=0)
            lrt_sv = pd.read_csv(os.path.join(self.data_dir, "fares", "lrt1_sv.csv"), index_col=0)

            # Store fare tables – keep both LRT variants
            self.fare_tables = {
                'Bus': bus_fares,
                'Jeep': jeep_fares,
                'LRT_REG': lrt_sj,       # Regular / single-journey
                'LRT_DISCOUNTED': lrt_sv  # Discounted / stored-value
            }
            self.logger.info(
                f"Loaded fare data: Bus ({len(bus_fares)} rows), Jeep ({len(jeep_fares)} rows), "
                f"LRT SJ ({len(lrt_sj)} stations), LRT SV ({len(lrt_sv)} stations)"
            )
        except Exception as e:
            self.logger.error(f"Failed to load fare data: {e}")
            self.fare_tables = {}

    def _count_transfers(self, segments):
        """Count the number of transfers in a route"""
        if len(segments) <= 1:
            return 0
        transfers = 0
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]
            # Count as transfer if different modes or different routes
            if (current_segment.mode != next_segment.mode or 
                current_segment.route_id != next_segment.route_id):
                transfers += 1
        return transfers

    def find_route(self, origin_stop: str, dest_stop: str, mode: str = 'fastest', prefer_transit: bool = True, max_transfers: int = 3):
        """Find a route between two stops using A* search (wrapper for find_route_astar) with mode-specific cost function."""
        cost_func = make_cost_function(mode)
        route_result = find_route_astar(
            origin_stop,
            dest_stop,
            prefer_transit,
            max_transfers,
            self.complete_graph,
            self.stops,
            self.osmnx_graph,
            self.logger,
            self._build_route_from_path,
            route_type=mode,
            cost_func=cost_func
        )
        if not route_result or not hasattr(route_result, 'segments'):
            return None
        class RouteResult:
            def __init__(self, segments):
                self.segments = segments
        return RouteResult(route_result.segments)

    def calculate_fare(self, mode: str, distance: float, fare_type: str = 'regular') -> float:
        """Calculate fare for a segment using fare_utils.calculate_fare."""
        try:
            return calculate_fare(mode, distance, fare_type)
        except Exception as e:
            self.logger.warning(f"Fare calculation failed for mode={mode}, distance={distance}: {e}")
            return 0.0

    # Alias used by earlier code section
    def calculate_real_fare(self, mode: str, from_stop: str, to_stop: str, distance: float, fare_type: str = 'regular') -> float:
        """Compute segment fare using loaded CSV tables (Bus/Jeep) or the LRT matrix.

        Fallbacks to the legacy distance buckets from ``fare_utils`` if the
        tabular lookup fails for whatever reason.  Supports the two fare
        columns that currently exist (``regular`` / ``discounted``)."""
        try:
            import math
            import pandas as pd  # Local import – only used if we actually look up the tables

            mode_norm = (mode or '').capitalize()

            # 1) Walking is always free
            if mode_norm == 'Walking':
                return 0.0

            if not getattr(self, 'fare_tables', None):
                # Tables not loaded – fallback to rough buckets
                return self.calculate_fare(mode_norm, distance, fare_type)

            # 2) Distance-based tables for Bus / Jeep
            if mode_norm in ['Bus', 'Jeep']:
                table = self.fare_tables.get(mode_norm)
                if table is not None and not table.empty:
                    km = max(1, math.ceil(distance))
                    # Find the first row whose distance >= travelled km
                    row = table[table['distance'] >= km].head(1)
                    if not row.empty:
                        column = 'regular' if fare_type != 'discounted' else 'discounted'
                        return float(row.iloc[0][column])

            # 3) Station-matrix for LRT (two variants)
            if mode_norm == 'Lrt':
                lrt_df = self.fare_tables.get('LRT_DISCOUNTED' if fare_type == 'discounted' else 'LRT_REG')
                if lrt_df is not None and not lrt_df.empty:
                    from_name = self.stops.get(from_stop, {}).get('name')
                    to_name = self.stops.get(to_stop, {}).get('name')
                    if from_name in lrt_df.index and to_name in lrt_df.columns:
                        fare_val = lrt_df.loc[from_name, to_name]
                        if pd.notna(fare_val):
                            return float(fare_val)
                # Fallback flat fare if lookup fails
                return 15.0

            # 4) Fallback to the simple heuristic if everything else failed
            return self.calculate_fare(mode_norm, distance, fare_type)

        except Exception as e:
            self.logger.warning(f"calculate_real_fare failed for {mode}:{from_stop}->{to_stop} ({distance:.2f} km): {e}")
            return self.calculate_fare(mode, distance, fare_type)

    # ---------------------------------------------------------------------
    #  NEW: Public helper expected by Flask API – multi-criteria routing
    # ---------------------------------------------------------------------
    def find_all_routes_with_coordinates(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        fare_type: str = 'regular',
        preferences: List[str] = None,
        allowed_modes: List[str] = None,
    ) -> Dict[str, Any]:
        """Return a dict {pref: route_dict} for each requested preference.

        This method is a lightweight wrapper around `find_route_with_walking` so
        that legacy Flask endpoints keep working without code changes.
        """
        if preferences is None:
            preferences = ['fastest']

        results: Dict[str, Any] = {}
        for pref in preferences:
            try:
                route = self.find_route_with_walking(
                    start_lat,
                    start_lon,
                    end_lat,
                    end_lon,
                    mode=pref,
                    fare_type=fare_type,
                )
                # Optionally filter by allowed_modes (front-end sends human names: "jeepney", "bus", …)
                if allowed_modes is not None and route and route.get('segments'):
                    def _canon(m: str) -> str:
                        m = (m or '').lower()
                        syn = {
                            'jeepney': 'jeep',
                            'jeep': 'jeep',
                            'fx': 'jeep',
                            'lrt': 'lrt',
                            'rail': 'lrt',
                            'train': 'lrt',
                            'tram': 'lrt',
                            'bus': 'bus',
                            'walk': 'walking',
                            'walking': 'walking',
                            'tricycle': 'tricycle',  # keep as-is even if we rarely generate it
                        }
                        return syn.get(m, m)

                    allowed_set = {_canon(m) for m in allowed_modes}
                    # Always keep walking and tricycle so first/last-mile helpers are not dropped
                    allowed_set.update({'walking', 'tricycle'})

                    filtered = [seg for seg in route['segments'] if _canon(seg.get('mode')) in allowed_set]

                    # If filtering removed everything, keep the original list to avoid empty routes
                    if filtered:
                        route['segments'] = filtered
                results[pref] = route or {}
            except Exception as e:
                self.logger.warning(f"find_all_routes_with_coordinates({pref}) failed: {e}")
                results[pref] = {}
        return results

    def _load_geojson_linestrings(self):
        """
        Load and index all LineStrings from fullcityofmanila.geojson (for street-based segments), lrtroutes.geojson (for LRT), and all files in data/routes-geojson (for route-specific fallbacks). Store as self.street_lines, self.lrt_lines, and self.route_lines.
        """
        import json
        import os
        from shapely.geometry import shape
        def load_linestrings_from_geojson(path):
            lines = []
            if not os.path.exists(path):
                self.logger.warning(f"GeoJSON file not found: {path}")
                return lines
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feature in data.get('features', []):
                geom = feature.get('geometry', {})
                if geom.get('type') == 'LineString':
                    lines.append({
                        'geometry': shape(geom),
                        'properties': feature.get('properties', {})
                    })
            return lines
        # Load street-based LineStrings
        street_geojson = os.path.join(self.data_dir, 'fullcityofmanila.geojson')
        self.street_lines = load_linestrings_from_geojson(street_geojson)
        self.logger.info(f"Loaded {len(self.street_lines)} street LineStrings from {street_geojson}")
        # Load LRT LineStrings
        lrt_geojson = os.path.join(self.data_dir, 'lrtroutes.geojson')
        self.lrt_lines = load_linestrings_from_geojson(lrt_geojson)
        self.logger.info(f"Loaded {len(self.lrt_lines)} LRT LineStrings from {lrt_geojson}")
        # Load all route-specific geojsons
        routes_geojson_dir = os.path.join(self.data_dir, 'routes-geojson')
        self.route_lines = {}
        if os.path.exists(routes_geojson_dir):
            for fname in os.listdir(routes_geojson_dir):
                if fname.endswith('.geojson'):
                    fpath = os.path.join(routes_geojson_dir, fname)
                    self.route_lines[fname] = load_linestrings_from_geojson(fpath)
        self.logger.info(f"Loaded {sum(len(v) for v in self.route_lines.values())} route LineStrings from {routes_geojson_dir}") 
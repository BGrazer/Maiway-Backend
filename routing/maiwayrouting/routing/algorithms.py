import networkx as nx
import osmnx as ox
from typing import List, Optional
from ..utils.geo_utils import haversine_distance
from ..models.route_segments import CompleteRoute
from functools import lru_cache

# === Multi-modal A* parameters (tune as needed) ===
FASTEST_MODE_SPEED_KMPH = 40.0  # LRT speed (km/h)
WALK_SPEED_KMPH = 5.0
JEEP_SPEED_KMPH = 20.0
BUS_SPEED_KMPH = 15.0
TRANSFER_PENALTY_MIN = 120.0  # Updated: strong pain for transfers (2 hours)

# Add these constants for mode-specific transfer penalties
LRT_TRANSFER_PENALTY = 4.0      # Lower penalty for LRT transfers (minutes)
BUS_TRANSFER_PENALTY = 15.0     # Higher penalty for bus transfers
JEEP_TRANSFER_PENALTY = 15.0    # Higher penalty for jeep transfers
WALK_TRANSFER_PENALTY = 20.0    # Highest penalty for walking transfers

DEFAULT_TRANSFER_PENALTY = TRANSFER_PENALTY_MIN  # Fallback/default

# Map mode to speed (km/h)
MODE_SPEEDS = {
    'lrt': FASTEST_MODE_SPEED_KMPH,
    'jeep': JEEP_SPEED_KMPH,
    'bus': BUS_SPEED_KMPH,
    'walk': WALK_SPEED_KMPH,
}

# Mode priorities for each route type (lower index = higher priority)
MODE_PRIORITIES = {
    'fastest':   ['lrt', 'jeepney', 'bus', 'walking'],
    'convenient':['bus', 'lrt', 'jeepney', 'walking'],
    'cheapest': ['jeepney', 'walking', 'bus', 'lrt'],
}

# Fare weights for each mode (for cheapest route)
MODE_FARES = {
    'lrt': 15.0,   # Example fare in PHP
    'jeep': 12.0,
    'bus': 13.0,
    'walk': 0.0,
}

WALKING_PENALTY = 1000.0  # Discourage mid-route walking unless unavoidable
FARE_DIVISOR = 10.0     # Fare has less influence during fastest/convenient searches

# Extreme cost applied to emergency straight-line walking fallback edges
FALLBACK_WALKING_PENALTY = 1500.0  # Extra pain for connectivity-fix walking edges
# Mode weights for different preferences (tuned)
mode_weights = {
    'fastest': {
        'lrt': 0.5,  # Strong bias towards LRT for fastest preference
        'bus': 2.0,  # Heavier cost to de-prioritise buses
        'jeepney': 2.5,
        'jeep': 2.5,
        'walking': 4.0,
        'walk': 4.0,
    },
    'cheapest': {
        'lrt': 2.0,      # Lowered to make LRT more competitive
        'bus': 2.0,      # Lowered
        'jeepney': 1.0,
        'jeep': 1.0,
        'walking': 0.5,
        'walk': 0.5,
    },
    'convenient': {
        'lrt': 1.2,      # Lowered to make LRT more competitive
        'bus': 1.0,
        'jeepney': 2.0,
        'jeep': 2.0,
        'walking': 3.0,
        'walk': 3.0,
    }
}

def heuristic(node: str, goal: str, prefer_transit: bool, stops: dict, osmnx_graph, logger) -> float:
    """A* heuristic function with OSMnx graph support"""
    if node not in stops or goal not in stops:
        return 0.0
    
    # Get coordinates
    node_lat, node_lon = stops[node]['lat'], stops[node]['lon']
    goal_lat, goal_lon = stops[goal]['lat'], stops[goal]['lon']
    
    # If we have OSMnx graph, use network distance as heuristic
    if osmnx_graph is not None:
        try:
            # Find nearest OSMnx nodes
            node_osmnx = ox.nearest_nodes(osmnx_graph, node_lon, node_lat)
            goal_osmnx = ox.nearest_nodes(osmnx_graph, goal_lon, goal_lat)
            
            # Calculate network distance using OSMnx
            try:
                network_distance = nx.shortest_path_length(
                    osmnx_graph, 
                    node_osmnx, 
                    goal_osmnx, 
                    weight='length'
                ) / 1000.0  # Convert to km
                
                # Use network distance as heuristic (more accurate than haversine)
                if prefer_transit:
                    return network_distance * 1.1  # Slight penalty for walking-heavy routes
                return network_distance
                
            except nx.NetworkXNoPath:
                # Fallback to haversine if no path in OSMnx graph
                pass
                
        except Exception as e:
            logger.debug(f"OSMnx heuristic failed: {e}, falling back to haversine")
    
    # Fallback to haversine distance
    distance = haversine_distance(node_lat, node_lon, goal_lat, goal_lon)
    
    # If prefer_transit, add penalty for walking-heavy routes
    if prefer_transit:
        distance *= 1.2  # Slight penalty for walking-heavy routes
    
    return distance

def get_mode_for_edge(u, v, complete_graph):
    data = complete_graph.get_edge_data(u, v)
    if data is None:
        return 'walk'
    return data.get('mode', 'walk')

def multimodal_heuristic(node: str, goal: str, stops: dict, complete_graph, osmnx_graph=None, route_type='fastest') -> float:
    """Distance-based time heuristic with optional OSMnx network guidance."""
    if node not in stops or goal not in stops:
        return 0.0

    node_lat, node_lon = stops[node]['lat'], stops[node]['lon']
    goal_lat, goal_lon = stops[goal]['lat'], stops[goal]['lon']

    # --- Fast (cached) network-distance heuristic via OSMnx ---
    if osmnx_graph is not None:
        try:
            # Cache nearest-node look-ups per stop id to avoid repeated KD-tree queries
            if not hasattr(multimodal_heuristic, "_nearest_cache"):
                multimodal_heuristic._nearest_cache = {}
            nearest_cache = multimodal_heuristic._nearest_cache

            def _nearest(lat, lon):
                key = (lat, lon)
                if key in nearest_cache:
                    return nearest_cache[key]
                nearest_cache[key] = ox.nearest_nodes(osmnx_graph, lon, lat)
                return nearest_cache[key]

            node_osm = _nearest(node_lat, node_lon)
            goal_osm = _nearest(goal_lat, goal_lon)

            # Cache expensive shortest_path_length calls with an LRU decorator
            if not hasattr(multimodal_heuristic, "_network_len"):
                @lru_cache(maxsize=50000)
                def _network_len(n1, n2):
                    return nx.shortest_path_length(osmnx_graph, n1, n2, weight="length")
                multimodal_heuristic._network_len = _network_len
            network_dist_km = multimodal_heuristic._network_len(node_osm, goal_osm) / 1000.0

            fastest_mode = MODE_PRIORITIES[route_type][0]
            speed = MODE_SPEEDS.get(fastest_mode, FASTEST_MODE_SPEED_KMPH)
            return network_dist_km / (speed / 60.0)
        except Exception:
            # Any failure → fall back to haversine
            pass

    # Straight-line (haversine) fallback
    distance_km = haversine_distance(node_lat, node_lon, goal_lat, goal_lon)
    fastest_mode = MODE_PRIORITIES[route_type][0]
    speed = MODE_SPEEDS.get(fastest_mode, FASTEST_MODE_SPEED_KMPH)
    return distance_km / (speed / 60.0)

def transfer_aware_weight(u, v, data, prev_mode=None, route_type='fastest'):
    """Compute edge cost factoring in distance, fare, walking/transfer penalties, and user preference (fastest / cheapest / convenient)."""
    # Extract edge attributes
    mode = str(data.get('mode', 'walking')).lower()
    distance = float(data.get('distance', 1.0))  # km
    fare = float(data.get('fare', 0.0))          # PHP

    # ---------------- Base distance cost ----------------
    weights = mode_weights.get(route_type, mode_weights['fastest'])
    mode_weight = weights.get(mode, weights.get('walking', 4.0))
    base_cost = distance * mode_weight

    # ---------------- Fare contribution ----------------
    fare_cost = fare if route_type == 'cheapest' else fare / FARE_DIVISOR

    # ---------------- Mode-specific tweaks ----------------
    if mode == 'lrt':
        # Encourage using LRT for longer hops
        if distance > 8:
            base_cost *= 0.6
        elif distance > 6:
            base_cost *= 0.7

    # Guard against obviously bad fare values
    if fare > 200:  # ¯\\_(ツ)_/¯ unrealistic – treat as prohibitive
        base_cost += 10_000

    # Preference-dependent multiplier for transfer pain
    transfer_mult = 1.5 if route_type == 'convenient' else 1.0

    # ---------------- Edge-level transfer penalties ----------------
    edge_type = str(data.get('type', '')).lower()
    if edge_type in {'transfer', 'walking_fallback'}:
        if data.get('is_lrt_transfer'):
            base_cost += (LRT_TRANSFER_PENALTY / 2) * transfer_mult
        else:
            per_mode_penalty = {
                'lrt': LRT_TRANSFER_PENALTY,
                'bus': BUS_TRANSFER_PENALTY,
                'jeep': JEEP_TRANSFER_PENALTY,
                'jeepney': JEEP_TRANSFER_PENALTY,
                'walk': WALK_TRANSFER_PENALTY,
                'walking': WALK_TRANSFER_PENALTY,
            }.get(mode, DEFAULT_TRANSFER_PENALTY)
            base_cost += per_mode_penalty * transfer_mult

    # Extra cost for fallback walking edges (graph connectivity hacks)
    if edge_type == 'walking_fallback':
        base_cost += FALLBACK_WALKING_PENALTY

    # ---------------- Walking deterrent ----------------
    if mode in {'walk', 'walking'} and distance > 0.2 and not data.get('is_lrt_transfer'):
        if route_type == 'cheapest':
            base_cost += WALKING_PENALTY / 2 + (distance * 5.0)
        else:
            base_cost += WALKING_PENALTY + (distance * 2.0)

    # ---------------- Mode-change penalty ----------------
    if prev_mode is not None and mode != prev_mode:
        change_penalty = {
            'lrt': LRT_TRANSFER_PENALTY,
            'bus': BUS_TRANSFER_PENALTY,
            'jeep': JEEP_TRANSFER_PENALTY,
            'jeepney': JEEP_TRANSFER_PENALTY,
            'walk': WALK_TRANSFER_PENALTY,
            'walking': WALK_TRANSFER_PENALTY,
        }.get(mode, DEFAULT_TRANSFER_PENALTY)
        base_cost += change_penalty * transfer_mult

    return base_cost + fare_cost

def bidirectional_astar(complete_graph, origin, dest, stops, osmnx_graph=None, route_type='fastest', cost_func=None):
    # True bidirectional A* with heuristics and transfer-aware weights
    # Pre-compute a cheap distance ratio once, so the heuristic per node is fast.
    origin_lat, origin_lon = stops[origin]['lat'], stops[origin]['lon']
    dest_lat, dest_lon = stops[dest]['lat'], stops[dest]['lon']

    # Straight-line distance between origin and destination (km)
    straight_origin_dest = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)

    network_ratio = 1.0  # default: network ~= straight-line
    if osmnx_graph is not None and straight_origin_dest > 0:
        try:
            o_node = ox.nearest_nodes(osmnx_graph, origin_lon, origin_lat)
            d_node = ox.nearest_nodes(osmnx_graph, dest_lon, dest_lat)
            network_len_km = nx.shortest_path_length(osmnx_graph, o_node, d_node, weight='length') / 1000.0
            # Cap ratio to avoid extreme overestimation (e.g. road detours around rivers)
            network_ratio = max(1.0, min(network_len_km / straight_origin_dest, 4.0))
        except Exception:
            # Any failure → keep ratio = 1.0 (fallback to haversine)
            network_ratio = 1.0

    # Make a fast heuristic closure that uses only haversine * precomputed ratio
    fastest_mode = MODE_PRIORITIES[route_type][0]
    speed_kmph = MODE_SPEEDS.get(fastest_mode, FASTEST_MODE_SPEED_KMPH)

    def fast_heuristic(node, goal):
        if node not in stops or goal not in stops:
            return 0.0
        n_lat, n_lon = stops[node]['lat'], stops[node]['lon']
        hav_km = haversine_distance(n_lat, n_lon, dest_lat, dest_lon)
        return (hav_km * network_ratio) / (speed_kmph / 60.0)

    try:
        def edge_weight(u, v, data):
            if cost_func:
                return cost_func(u, v, data)
            return transfer_aware_weight(u, v, data, route_type=route_type)
        # Forward A*
        path1 = nx.astar_path(
            complete_graph, origin, dest,
            weight=edge_weight,
            heuristic=fast_heuristic
        )
        # Reverse A*
        path2 = nx.astar_path(
            complete_graph, dest, origin,
            weight=edge_weight,
            heuristic=fast_heuristic
        )
        if len(path1) <= len(path2):
            return path1
        else:
            return list(reversed(path2))
    except Exception:
        return []
    finally:
        # Clear per-search caches to keep memory usage bounded
        if hasattr(multimodal_heuristic, "_nearest_cache"):
            multimodal_heuristic._nearest_cache.clear()
        if hasattr(multimodal_heuristic, "_network_len") and hasattr(multimodal_heuristic._network_len, "cache_clear"):
            multimodal_heuristic._network_len.cache_clear()

def find_route_astar(origin: str, dest: str, prefer_transit: bool, max_transfers: int, 
                    complete_graph, stops: dict, osmnx_graph, logger, build_route_from_path_func, route_type='fastest', cost_func=None) -> Optional[CompleteRoute]:
    """Find route using bidirectional A* with multi-modal, transfer-aware heuristic and mode priorities, supporting custom cost_func."""
    if origin not in complete_graph or dest not in complete_graph:
        logger.warning(f"Origin {origin} or destination {dest} not in complete graph")
        return None
    try:
        if not nx.has_path(complete_graph, origin, dest):
            logger.warning(f"No path exists between {origin} and {dest}")
            return None
        # Use bidirectional A*
        path = bidirectional_astar(complete_graph, origin, dest, stops, osmnx_graph=osmnx_graph, route_type=route_type, cost_func=cost_func)
        if not path:
            logger.warning(f"Bidirectional A* returned empty path between {origin} and {dest}")
            return None
        logger.info(f"Bidirectional A* found path: {len(path)} nodes from {origin} to {dest}")
        logger.debug(f"Path found: {path}")
        for i in range(len(path)-1):
            edge = complete_graph.get_edge_data(path[i], path[i+1])
            logger.debug(f"Edge {path[i]}->{path[i+1]}: {edge}")
        return build_route_from_path_func(path)
    except nx.NetworkXNoPath:
        logger.warning(f"No path exists between {origin} and {dest} (NetworkXNoPath)")
        return None
    except Exception as e:
        logger.error(f"Bidirectional A* routing failed: {e}")
        return None

def find_route_astar_with_osmnx(origin: str, dest: str, prefer_transit: bool, max_transfers: int,
                               stops: dict, osmnx_graph, complete_graph, logger) -> List[str]:
    """Enhanced A* algorithm that considers Manila OSMnx graph for better routing"""
    if origin not in stops or dest not in stops:
        return []
    
    # Get coordinates
    origin_lat, origin_lon = stops[origin]['lat'], stops[origin]['lon']
    dest_lat, dest_lon = stops[dest]['lat'], stops[dest]['lon']
    
    # Find nearest OSMnx nodes
    origin_osmnx = ox.nearest_nodes(osmnx_graph, origin_lon, origin_lat)
    dest_osmnx = ox.nearest_nodes(osmnx_graph, dest_lon, dest_lat)
    
    # Calculate network distance using OSMnx for better pathfinding
    try:
        # Get the actual network path through Manila roads
        osmnx_path = nx.shortest_path(
            osmnx_graph, 
            origin_osmnx, 
            dest_osmnx, 
            weight='length'
        )
        
        # Convert OSMnx path to network distance
        network_distance = nx.shortest_path_length(
            osmnx_graph,
            origin_osmnx,
            dest_osmnx,
            weight='length'
        ) / 1000.0  # Convert to km
        
        logger.info(f"OSMnx network path: {len(osmnx_path)} nodes, {network_distance:.3f} km")
        
        # Use this network distance to influence A* routing
        # Prefer routes that follow Manila road network
        return astar_with_network_constraint(origin, dest, network_distance, prefer_transit, max_transfers, stops, osmnx_graph, complete_graph)
        
    except nx.NetworkXNoPath:
        # Fallback to basic A* if no OSMnx path
        logger.warning("No OSMnx path found, using basic A*")
        return nx.astar_path(complete_graph, origin, dest, weight='weight')

def astar_with_network_constraint(origin: str, dest: str, network_distance: float, prefer_transit: bool, max_transfers: int,
                                 stops: dict, osmnx_graph, complete_graph) -> List[str]:
    """A* algorithm that considers network distance constraint"""
    # Use network distance as a constraint for better routing
    # This ensures routes follow Manila road network more closely
    
    # Custom heuristic that considers network distance
    def network_heuristic(node: str, goal: str) -> float:
        if node not in stops or goal not in stops:
            return 0.0
        
        # Get coordinates
        node_lat, node_lon = stops[node]['lat'], stops[node]['lon']
        goal_lat, goal_lon = stops[goal]['lat'], stops[goal]['lon']
        
        # Use network distance as heuristic
        try:
            node_osmnx = ox.nearest_nodes(osmnx_graph, node_lon, node_lat)
            goal_osmnx = ox.nearest_nodes(osmnx_graph, goal_lon, goal_lat)
            
            heuristic_distance = nx.shortest_path_length(
                osmnx_graph,
                node_osmnx,
                goal_osmnx,
                weight='length'
            ) / 1000.0
            
            # Add penalty for routes that deviate too much from network distance
            if prefer_transit:
                heuristic_distance *= 1.1  # Slight penalty for walking-heavy routes
            
            return heuristic_distance
            
        except nx.NetworkXNoPath:
            # Fallback to haversine
            return haversine_distance(node_lat, node_lon, goal_lat, goal_lon)
    
    # Use NetworkX A* with custom heuristic
    return nx.astar_path(
        complete_graph, 
        origin, 
        dest, 
        weight='weight',
        heuristic=network_heuristic
    ) 

# === Wrapper to get all three route types ===
def find_all_route_types(origin, dest, complete_graph, stops, osmnx_graph, logger, build_route_from_path_func):
    """Return fastest, most convenient, and cheapest routes as CompleteRoute objects using bidirectional A*"""
    results = {}
    for route_type in ['fastest', 'convenient', 'cheapest']:
        route = find_route_astar(
            origin, dest, True, 5, complete_graph, stops, osmnx_graph, logger, build_route_from_path_func, route_type=route_type
        )
        results[route_type] = route
    return results 

# ---------------------------------------------------------------------------
#  Lightweight Yen k-shortest paths (edge-disjoint allowed) – tuned for MaiWay
# ---------------------------------------------------------------------------
from heapq import heappush, heappop

def yen_k_shortest_paths(G, source: str, target: str, k: int, weight_key: str = 'weight'):
    """Return up to *k* simple paths (as lists of nodes) ordered by total *weight_key*.

    Very small, dependency-free implementation adequate for graphs of 
    a few thousand nodes.  Based on NetworkX reference but inlined to
    avoid bringing large helper modules.
    """
    import networkx as nx

    if source == target:
        return [[source]]

    # First shortest path (Dijkstra)
    try:
        length, path = nx.single_source_dijkstra(G, source, target, weight=weight_key)
    except nx.NetworkXNoPath:
        return []
    A = [path]            # List of shortest paths found so far
    A_len = [length]      # Their lengths
    B = []                # Min-heap of candidate paths

    for k_i in range(1, k):
        for i in range(len(A[-1]) - 1):
            spur_node = A[-1][i]
            root_path = A[-1][:i + 1]

            # Remove the edges / nodes that would create loops or already-seen prefixes
            removed_edges = []
            removed_nodes = []
            for p in A:
                if len(p) > i and p[:i + 1] == root_path:
                    u = p[i]
                    v = p[i + 1]
                    if G.has_edge(u, v):
                        attr = G[u][v]
                        removed_edges.append((u, v, attr))
                        G.remove_edge(u, v)
            for n in root_path[:-1]:  # exclude spur node
                if G.has_node(n):
                    # store all incident edges so we can restore quickly
                    incident = list(G.edges(n, data=True))
                    removed_nodes.append((n, incident))
                    G.remove_node(n)

            try:
                spur_len, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight_key)
                total_path = root_path[:-1] + spur_path
                total_len = sum(G[u][v][weight_key] for u, v in zip(total_path[:-1], total_path[1:]))
                heappush(B, (total_len, total_path))
            except nx.NetworkXNoPath:
                pass  # No spur path found
            # Restore graph
            for u, v, attr in removed_edges:
                G.add_edge(u, v, **attr)
            for n, incident in removed_nodes:
                G.add_node(n)
                for u, v, attr in incident:
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, **attr)
        if not B:
            break
        length, path = heappop(B)
        A.append(path)
        A_len.append(length)
    return A[:k] 
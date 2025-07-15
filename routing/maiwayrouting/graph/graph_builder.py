import os
import pickle
import time
from multiprocessing import Pool, cpu_count
import networkx as nx
import numpy as np
from rtree import index
from ..utils.geo_utils import haversine_distance
from ..models.route_segments import TransitSegment

# You may want to wrap these in a class or keep as functions depending on usage.

def vectorized_haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation using numpy"""
    R = 6371000  # Earth's radius in meters
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def build_transit_graph(stops, routes, trips, stop_times, mode_weights, logger):
    """Build transit graph with ONLY consecutive stops in same trip (GTFS compliant)"""
    transit_graph = nx.DiGraph()
    # Group stop times by trip
    trip_stop_times = {}
    for stop_time in stop_times:
        trip_id = stop_time['trip_id']
        if trip_id not in trip_stop_times:
            trip_stop_times[trip_id] = []
        trip_stop_times[trip_id].append(stop_time)
    # Sort by stop sequence within each trip
    for trip_id in trip_stop_times:
        trip_stop_times[trip_id].sort(key=lambda x: x['stop_sequence'])
    # Create transit edges ONLY between consecutive stops in same trip
    transit_edges = 0
    for trip_id, stop_times_list in trip_stop_times.items():
        if trip_id not in trips:
            logger.warning(f"Trip {trip_id} not found in trips")
            continue
        trip = trips[trip_id]
        route = routes.get(trip['route_id'])
        if not route:
            logger.warning(f"Route {trip['route_id']} not found for trip {trip_id}")
            continue
        # --- Restore correct GTFS route_type to mode mapping ---
        route_type_val = route.get('route_type', 3)
        if route_type_val == 0:
            mode = 'LRT'
        elif route_type_val == 3:
            mode = 'Bus'
        elif 200 <= route_type_val < 300:
            mode = 'Jeep'
        else:
            # Heuristic fallback based on route_id / name
            rl = (route.get('route_long_name', '') + route.get('route_short_name', '')).lower()
            if 'lrt' in rl or 'line' in rl:
                mode = 'LRT'
            elif 'bus' in rl:
                mode = 'Bus'
            elif 'jeep' in rl or 'jp' in rl:
                mode = 'Jeep'
            else:
                mode = 'Bus'
        
        # Debug: Log trip info
        logger.debug(f"Processing trip {trip_id}: {len(stop_times_list)} stops")
        
        for i in range(len(stop_times_list) - 1):
            current_stop_time = stop_times_list[i]
            next_stop_time = stop_times_list[i + 1]
            
            # Check if stops exist
            if current_stop_time['stop_id'] not in stops:
                logger.warning(f"Stop {current_stop_time['stop_id']} not found for trip {trip_id}")
                continue
            if next_stop_time['stop_id'] not in stops:
                logger.warning(f"Stop {next_stop_time['stop_id']} not found for trip {trip_id}")
                continue
            
            current_stop = stops[current_stop_time['stop_id']]
            next_stop = stops[next_stop_time['stop_id']]
            distance = haversine_distance(
                current_stop['lat'], current_stop['lon'],
                next_stop['lat'], next_stop['lon']
            )
            edge_data = {
                'trip_id': trip_id,
                'route_id': trip['route_id'],
                'mode': mode,
                'distance': distance,
                'weight': distance * mode_weights.get(mode, 1.0),
                'stop_sequence_from': current_stop_time['stop_sequence'],
                'stop_sequence_to': next_stop_time['stop_sequence'],
                'type': 'transit'
            }
            # Add edge in correct direction (current → next)
            transit_graph.add_edge(
                current_stop_time['stop_id'],
                next_stop_time['stop_id'],
                **edge_data
            )
            transit_edges += 1
            
            # Debug: Log first few edges for route_001
            if trip_id in ['route_001_inbound', 'route_001_outbound'] and i < 5:
                logger.debug(f"  Edge {i+1}: {current_stop_time['stop_id']} → {next_stop_time['stop_id']} (seq {current_stop_time['stop_sequence']}→{next_stop_time['stop_sequence']})")
    
    logger.info(f"Transit graph built: {transit_graph.number_of_nodes()} nodes, {transit_graph.number_of_edges()} edges")
    logger.info(f"Created {transit_edges} transit edges")
    # --- Count edges by mode ---
    from collections import Counter
    mode_counter = Counter()
    lrt_edges = []
    for u, v, data in transit_graph.edges(data=True):
        mode = data.get('mode', 'unknown')
        mode_counter[mode] += 1
        if mode == 'LRT' and len(lrt_edges) < 10:
            lrt_edges.append((u, v, data))
    logger.info(f"Transit edge counts by mode: {dict(mode_counter)}")
    if lrt_edges:
        logger.info(f"First few LRT edges: {lrt_edges}")

    # --- Add LRT ride edges from lrt_edges.csv ---
    import pandas as pd
    lrt_edges_path = os.path.join(os.getcwd(), 'data', 'lrt_edges.csv')
    if os.path.exists(lrt_edges_path):
        lrt_df = pd.read_csv(lrt_edges_path)
        added = 0
        for _, row in lrt_df.iterrows():
            from_stop = row['from_stop']
            to_stop = row['to_stop']
            # Convert stored metre distance to kilometres for consistency
            distance = row['distance_m'] / 1000.0
            # Compute weight freshly to avoid unit issues
            weight = distance * mode_weights.get('lrt', 1.0)
            edge_data = {
                'trip_id': 'LRT_SHAPE',
                'route_id': 'LRT_SHAPE',
                'mode': 'lrt',
                'distance': distance,
                'weight': weight,
                'stop_sequence_from': None,
                'stop_sequence_to': None,
                'type': 'transit'
            }
            # Only add if not already present
            if not transit_graph.has_edge(from_stop, to_stop):
                transit_graph.add_edge(from_stop, to_stop, **edge_data)
                added += 1
            # Add reverse direction if absent
            if not transit_graph.has_edge(to_stop, from_stop):
                transit_graph.add_edge(to_stop, from_stop, **edge_data)
                added += 1
        logger.info(f"Added {added} LRT ride edges from data/lrt_edges.csv to transit graph.")
    else:
        logger.warning(f"data/lrt_edges.csv not found, skipping LRT shape edges integration.")

    return transit_graph

def build_smart_walking_edges(stops, stop_times, trips, routes, mode_weights, transfer_penalty, logger, walking_graph, allowed_transfers=None):
    """
    Build walking/transfer edges between stops. Sequentially build, then cache results. On future runs, load from cache.
    """
    cache_file = os.path.join("cache", "walking_edges_cache_v2.pkl")  # new version to invalidate old meter-based cache
    if os.path.exists(cache_file):
        logger.info("Loading walking/transfer edges from cache...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        walking_edges = cache_data['walking_edges']
        for (from_stop, to_stop, attrs) in walking_edges:
            walking_graph.add_edge(from_stop, to_stop, **attrs)
        logger.info(f"Loaded {len(walking_edges)} walking/transfer edges from cache.")
        return
    logger.info("Building walking/transfer edges (no parallelization)...")
    print(">>> Starting walking edge processing...", flush=True)
    start_time = time.time()
    all_stops = list(stops.keys())
    print(f">>> Found {len(all_stops)} total stops", flush=True)
    
    # Pre-compute LRT stops for faster lookup
    print(">>> Computing LRT stops...", flush=True)
    lrt_stops = set()
    for stop_id in all_stops:
        if is_lrt_stop(stop_id, stop_times, trips, routes):
            lrt_stops.add(stop_id)
    
    print(f">>> Found {len(lrt_stops)} LRT stops out of {len(all_stops)} total stops", flush=True)
    
    # Fast pre-filtering using vectorized operations
    print(">>> Fast pre-filtering stops to reduce pairs...", flush=True)
    
    # Convert to numpy arrays for vectorized operations
    stop_ids_array = np.array(all_stops)
    lats_array = np.array([stops[stop_id]['lat'] for stop_id in all_stops])
    lons_array = np.array([stops[stop_id]['lon'] for stop_id in all_stops])
    
    # Create a mask for LRT stops
    lrt_mask = np.array([stop_id in lrt_stops for stop_id in all_stops])
    
    # Always keep LRT stops
    keep_mask = lrt_mask.copy()
    
    # For non-LRT stops, check if they have any nearby stops within 800m
    non_lrt_indices = np.where(~lrt_mask)[0]
    print(f">>> Checking {len(non_lrt_indices)} non-LRT stops for nearby connections...", flush=True)
    
    # Process non-LRT stops in batches to avoid memory issues
    batch_size = 1000
    for batch_start in range(0, len(non_lrt_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(non_lrt_indices))
        batch_indices = non_lrt_indices[batch_start:batch_end]
        
        print(f">>> Processing batch {batch_start//batch_size + 1}/{(len(non_lrt_indices) + batch_size - 1)//batch_size}: stops {batch_start+1}-{batch_end}", flush=True)
        
        # For each stop in this batch, check distances to all other stops
        for idx in batch_indices:
            # Calculate distances from this stop to all other stops
            distances = vectorized_haversine(
                lats_array[idx], lons_array[idx],
                lats_array, lons_array
            )
            
            # Count stops within 800m (excluding self)
            nearby_count = np.sum((distances > 0) & (distances <= 800))
            
            # Keep if there's at least one nearby stop
            if nearby_count > 0:
                keep_mask[idx] = True
    
    # Apply the mask to get filtered stops
    filtered_stops = stop_ids_array[keep_mask].tolist()
    print(f">>> Pre-filtered from {len(all_stops)} to {len(filtered_stops)} stops", flush=True)
    all_stops = filtered_stops
    
    # Use R-tree for efficient spatial queries
    print(">>> Building R-tree for spatial indexing...", flush=True)
    p = index.Property()
    p.dimension = 2  # lat, lon
    # The R-tree index will store the index of the stop in all_stops
    idx = index.Index(properties=p)
    
    # Add all stops to the index
    for i, stop_id in enumerate(all_stops):
        lat = stops[stop_id]['lat']
        lon = stops[stop_id]['lon']
        # R-tree needs a bounding box: (min_lon, min_lat, max_lon, max_lat)
        # For a point, min and max are the same.
        idx.insert(i, (lon, lat, lon, lat))

    print(">>> R-tree built. Finding walking edges...", flush=True)
    
    results = []
    
    for i, stop_id in enumerate(all_stops):
        if (i + 1) % 100 == 0:
            print(f">>> Processing stop {i+1}/{len(all_stops)}... Found {len(results)} edges.", flush=True)

        lat = stops[stop_id]['lat']
        lon = stops[stop_id]['lon']
        
        is_lrt_stop_i = stop_id in lrt_stops
        max_dist_km = 2.0 if is_lrt_stop_i else 0.4
        
        # Approximate search radius in degrees. 1 degree lat ~ 111km.
        # This is a rough but fast way to define a bounding box for the query.
        search_radius_deg = max_dist_km / 111.0
        bounds = (lon - search_radius_deg, lat - search_radius_deg, lon + search_radius_deg, lat + search_radius_deg)
        
        # Query the R-tree for stops within the bounding box
        candidate_indices = list(idx.intersection(bounds))

        for j in candidate_indices:
            # Avoid self-loops and processing pairs twice
            if i >= j:
                continue

            neighbor_stop_id = all_stops[j]
            
            # Precise distance check
            neighbor_lat = stops[neighbor_stop_id]['lat']
            neighbor_lon = stops[neighbor_stop_id]['lon']
            
            dist_m = vectorized_haversine(lat, lon, neighbor_lat, neighbor_lon)
            dist_km = dist_m / 1000.0

            is_lrt_stop_j = neighbor_stop_id in lrt_stops
            is_lrt_connection = is_lrt_stop_i or is_lrt_stop_j
            current_max_dist = 2.0 if is_lrt_connection else 0.4

            if dist_km <= current_max_dist:
                attrs = {
                    'weight': transfer_penalty,
                    'mode': 'walk',
                    'distance': dist_km,
                    'type': 'transfer',
                    'is_lrt_transfer': is_lrt_connection
                }
                results.append((stop_id, neighbor_stop_id, attrs))
                results.append((neighbor_stop_id, stop_id, attrs))
    
    print(f">>> Finished processing! Found {len(results)} walking edges", flush=True)
    walking_edges = results
    for (from_stop, to_stop, attrs) in walking_edges:
        walking_graph.add_edge(from_stop, to_stop, **attrs)
    # Ensure cache directory exists
    os.makedirs('cache', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({'walking_edges': walking_edges}, f)
    elapsed = time.time() - start_time
    logger.info(f"Built and cached {len(walking_edges):,} walking/transfer edges in {elapsed:.2f} seconds")
    logger.info(f"Processing speed: {len(walking_edges)/elapsed:.1f} edges/second")

def build_complete_graph(transit_graph, walking_graph, logger):
    """Build complete graph combining transit and walking edges"""
    # Start with transit graph
    complete_graph = transit_graph.copy()
    
    # Add all nodes from walking graph that aren't in transit graph
    for node in walking_graph.nodes():
        if node not in complete_graph:
            complete_graph.add_node(node)
    
    # Add walking edges ONLY where transit edges don't exist
    walking_edges = 0
    overwritten_transit = 0
    
    for edge in walking_graph.edges(data=True):
        from_stop, to_stop, data = edge
        
        # Check if a transit edge already exists between these stops
        existing_edge = complete_graph.get_edge_data(from_stop, to_stop)
        if existing_edge and existing_edge.get('type') == 'transit':
            # Skip this walking edge - transit edge takes priority
            overwritten_transit += 1
            continue
        
        # Add walking edge
        complete_graph.add_edge(from_stop, to_stop, **data)
        walking_edges += 1
    
    logger.info(f"Complete graph built: {complete_graph.number_of_nodes()} nodes, {complete_graph.number_of_edges()} edges")
    logger.info(f"Transit edges: {transit_graph.number_of_edges()}")
    logger.info(f"Walking edges added: {walking_edges}")
    logger.info(f"Walking edges skipped (transit priority): {overwritten_transit}")
    logger.info(f"Total edges: {complete_graph.number_of_edges()}")
    
    return complete_graph

def is_useful_transfer(stop1, stop2, stop_times, trips, routes, route_types):
    """Check if walking between two stops creates a useful transfer"""
    routes1 = set()
    routes2 = set()
    for stop_time in stop_times:
        if stop_time['stop_id'] == stop1:
            trip_id = stop_time['trip_id']
            if trip_id in trips:
                route_id = trips[trip_id]['route_id']
                routes1.add(route_id)
                if route_id not in route_types:
                    route_types[route_id] = routes[route_id]['route_type']
        elif stop_time['stop_id'] == stop2:
            trip_id = stop_time['trip_id']
            if trip_id in trips:
                route_id = trips[trip_id]['route_id']
                routes2.add(route_id)
                if route_id not in route_types:
                    route_types[route_id] = routes[route_id]['route_type']
    if routes1.intersection(routes2):
        return False
    def get_ltfrb_type(route_id):
        if route_id.startswith('LTFRB_PUB'):
            return 'PUB'
        elif route_id.startswith('LTFRB_PUJ'):
            return 'PUJ'
        return None
    ltfrb_types1 = {get_ltfrb_type(r) for r in routes1 if get_ltfrb_type(r)}
    ltfrb_types2 = {get_ltfrb_type(r) for r in routes2 if get_ltfrb_type(r)}
    if ltfrb_types1 and ltfrb_types2 and ltfrb_types1 == ltfrb_types2:
        return False
    return len(routes1) > 0 and len(routes2) > 0

def is_lrt_stop(stop_id, stop_times, trips, routes):
    """Return True if stop_id is used by any trip on a route_type==0 (LRT) route."""
    for st in stop_times:
        if st['stop_id'] == stop_id:
            trip = trips.get(st['trip_id'])
            if trip:
                route = routes.get(trip['route_id'])
                if route and route.get('route_type', 3) == 0:
                    return True
    return False

# Similarly, move and adapt _build_smart_walking_edges and _build_complete_graph here. 
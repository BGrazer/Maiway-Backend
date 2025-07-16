#!/usr/bin/env python3
"""
Filter stops to reduce density while preserving important stops
- Keep every other stop
- Keep first and last stops of each route
- Keep ALL LRT stops
- Keep transfer points from transfers.txt
"""

import pandas as pd
import os
import shutil
from datetime import datetime

def identify_lrt_stops(stops_df, routes_df):
    """Identify LRT stops based on route information"""
    lrt_stops = set()
    
    # Look for LRT routes (route_type 0 or route_id containing 'lrt')
    lrt_routes = routes_df[
        (routes_df['route_type'] == 0) | 
        (routes_df['route_id'].str.contains('lrt', case=False, na=False))
    ]['route_id'].tolist()
    
    print(f"Found {len(lrt_routes)} LRT routes")
    
    # Get all stops that serve LRT routes
    if 'stop_times.txt' in os.listdir('routing_data'):
        stop_times_df = pd.read_csv('routing_data/stop_times.txt')
        trips_df = pd.read_csv('routing_data/trips.txt')
        
        # Get trips for LRT routes
        lrt_trips = trips_df[trips_df['route_id'].isin(lrt_routes)]['trip_id'].tolist()
        
        # Get stops for LRT trips
        lrt_stop_times = stop_times_df[stop_times_df['trip_id'].isin(lrt_trips)]
        lrt_stops = set(lrt_stop_times['stop_id'].unique())
        
        print(f"Found {len(lrt_stops)} LRT stops")
    
    return lrt_stops

def identify_transfer_stops():
    """Identify stops that are transfer points"""
    transfer_stops = set()
    
    if 'transfers.txt' in os.listdir('routing_data'):
        transfers_df = pd.read_csv('routing_data/transfers.txt')
        transfer_stops = set(transfers_df['from_stop_id'].unique()) | set(transfers_df['to_stop_id'].unique())
        print(f"Found {len(transfer_stops)} transfer stops")
    
    return transfer_stops

def identify_route_endpoints(stops_df):
    """Identify first and last stops of each route"""
    endpoint_stops = set()
    
    if 'stop_times.txt' in os.listdir('routing_data') and 'trips.txt' in os.listdir('routing_data'):
        stop_times_df = pd.read_csv('routing_data/stop_times.txt')
        trips_df = pd.read_csv('routing_data/trips.txt')
        
        # Group by trip and get first/last stops
        for trip_id in trips_df['trip_id'].unique():
            trip_stops = stop_times_df[stop_times_df['trip_id'] == trip_id].sort_values('stop_sequence')
            if len(trip_stops) > 0:
                first_stop = trip_stops.iloc[0]['stop_id']
                last_stop = trip_stops.iloc[-1]['stop_id']
                endpoint_stops.add(first_stop)
                endpoint_stops.add(last_stop)
        
        print(f"Found {len(endpoint_stops)} route endpoint stops")
    
    return endpoint_stops

def filter_stops():
    """Main filtering function"""
    print("FILTERING STOPS TO REDUCE DENSITY")
    print("=" * 50)
    
    # Load data
    stops_df = pd.read_csv('routing_data/stops.txt')
    routes_df = pd.read_csv('routing_data/routes.txt')
    
    print(f"Original stops: {len(stops_df)}")
    
    # Identify important stops to preserve
    lrt_stops = identify_lrt_stops(stops_df, routes_df)
    transfer_stops = identify_transfer_stops()
    endpoint_stops = identify_route_endpoints(stops_df)
    
    # Combine all important stops
    important_stops = lrt_stops | transfer_stops | endpoint_stops
    print(f"Important stops to preserve: {len(important_stops)}")
    
    # Group stops by route (assuming stop_id format: route_stopnumber)
    route_groups = {}
    for _, stop in stops_df.iterrows():
        stop_id = stop['stop_id']
        # Extract route from stop_id (e.g., "stop_1_001" -> "1")
        if '_' in stop_id:
            route_part = stop_id.split('_')[1] if len(stop_id.split('_')) > 1 else stop_id
            if route_part not in route_groups:
                route_groups[route_part] = []
            route_groups[route_part].append(stop_id)
    
    print(f"Found {len(route_groups)} route groups")
    
    # Filter stops: keep every other stop + important stops
    kept_stops = set()
    
    for route, stop_list in route_groups.items():
        # Sort stops by their number
        sorted_stops = sorted(stop_list, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        
        # Keep every other stop
        for i, stop_id in enumerate(sorted_stops):
            if i % 2 == 0:  # Keep even indices (0, 2, 4, ...)
                kept_stops.add(stop_id)
        
        # Always keep first and last stops of this route
        if len(sorted_stops) > 0:
            kept_stops.add(sorted_stops[0])  # First stop
            kept_stops.add(sorted_stops[-1])  # Last stop
    
    # Add all important stops
    kept_stops.update(important_stops)
    
    # Filter the stops dataframe
    filtered_stops = stops_df[stops_df['stop_id'].isin(kept_stops)]
    
    print(f"Filtered stops: {len(filtered_stops)}")
    print(f"Reduction: {((len(stops_df) - len(filtered_stops)) / len(stops_df) * 100):.1f}%")
    
    # Backup original and save filtered
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"routing_data/stops_backup_{timestamp}.txt"
    shutil.copy('routing_data/stops.txt', backup_file)
    print(f"Backup saved: {backup_file}")
    
    # Save filtered stops
    filtered_stops.to_csv('routing_data/stops.txt', index=False)
    print("Filtered stops saved to routing_data/stops.txt")
    
    # Also filter stop_times.txt
    if 'stop_times.txt' in os.listdir('routing_data'):
        stop_times_df = pd.read_csv('routing_data/stop_times.txt')
        filtered_stop_times = stop_times_df[stop_times_df['stop_id'].isin(kept_stops)]
        
        backup_stop_times = f"routing_data/stop_times_backup_{timestamp}.txt"
        shutil.copy('routing_data/stop_times.txt', backup_stop_times)
        print(f"Stop times backup saved: {backup_stop_times}")
        
        filtered_stop_times.to_csv('routing_data/stop_times.txt', index=False)
        print(f"Filtered stop times: {len(filtered_stop_times)} (from {len(stop_times_df)})")
    
    print("\nStop filtering completed!")
    print("Run 'python routing.py' to rebuild cache with filtered stops")

if __name__ == "__main__":
    filter_stops() 
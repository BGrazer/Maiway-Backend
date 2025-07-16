import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import os

def haversine(lat1, lon1, lat2, lon2):
    R = 6371_000  # meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# --- Load data ---
stops = pd.read_csv('routing_data/stops.txt', header=0)
stop_times = pd.read_csv('routing_data/stop_times.txt', header=0)
shapes = pd.read_csv('routing_data/shapes.txt', header=0)

# Ensure lat/lon are floats
shapes['lat'] = shapes.iloc[:,1].astype(float) if 'lat' not in shapes.columns else shapes['lat'].astype(float)
shapes['lon'] = shapes.iloc[:,2].astype(float) if 'lon' not in shapes.columns else shapes['lon'].astype(float)
stops['lat'] = stops.iloc[:,2].astype(float) if 'lat' not in stops.columns else stops['lat'].astype(float)
stops['lon'] = stops.iloc[:,3].astype(float) if 'lon' not in stops.columns else stops['lon'].astype(float)

# --- Filter for LRT route (example: route_095_outbound, shape_095) ---
lrt_trip_id = 'route_095_outbound'
lrt_shape_id = 'shape_095'

lrt_stop_times = stop_times[stop_times.iloc[:,0] == lrt_trip_id].sort_values(stop_times.columns[4])
lrt_stops = lrt_stop_times.iloc[:,3].tolist()
lrt_stops_df = stops[stops.iloc[:,0].isin(lrt_stops)].set_index(stops.columns[0])
shape_points = shapes[shapes.iloc[:,0] == lrt_shape_id].sort_values(shapes.columns[3])

def closest_shape_idx(lat, lon, shape_points):
    dists = ((shape_points['lat'] - lat)**2 + (shape_points['lon'] - lon)**2).values
    return np.argmin(dists)

stop_to_shape_idx = {}
for stop_id in lrt_stops:
    lat = float(lrt_stops_df.loc[stop_id, 'lat'])
    lon = float(lrt_stops_df.loc[stop_id, 'lon'])
    idx = closest_shape_idx(lat, lon, shape_points)
    stop_to_shape_idx[stop_id] = idx

edges = []
for i in range(len(lrt_stops) - 1):
    from_stop = lrt_stops[i]
    to_stop = lrt_stops[i+1]
    idx1 = stop_to_shape_idx[from_stop]
    idx2 = stop_to_shape_idx[to_stop]
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    segment = shape_points.iloc[idx1:idx2+1]
    dist = 0.0
    for j in range(len(segment)-1):
        lat1, lon1 = segment.iloc[j][['lat', 'lon']]
        lat2, lon2 = segment.iloc[j+1][['lat', 'lon']]
        dist += haversine(lat1, lon1, lat2, lon2)
    edge = {
        'from_stop': from_stop,
        'to_stop': to_stop,
        'mode': 'lrt',
        'distance_m': dist,
        'weight': dist / 30000,  # 30km/h LRT speed
    }
    edges.append(edge)

edges_df = pd.DataFrame(edges)
edges_df.to_csv('routing_data/lrt_edges.csv', index=False)
print(f"LRT ride edges saved to routing_data/lrt_edges.csv ({len(edges)} edges)") 
"""
UnifiedShapeGenerator: Combined shape generator with GTFS and Mapbox integration
Combines features from ShapeGenerator and EnhancedShapeGenerator
"""

import pandas as pd
import numpy as np
import math
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import sys

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class CoreShapeGenerator:

    """
    Unified shape generator that combines GTFS shape processing with Mapbox integration.
    Provides accurate route visualization using GTFS shapes when available,
    falls back to Mapbox API for missing shapes, and uses straight-line interpolation as last resort.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.stops_cache = {}  # Cache for stop coordinates
        # No GTFS shapes or trips loaded at all

    def set_stops_cache(self, stops: Dict[str, Any]):
        """Set stops cache for coordinate lookup"""
        self.stops_cache = stops
        logger.debug(f"Set stops cache with {len(stops)} stops")

    def get_shape_for_trip(self, trip_id: str) -> Optional[List[Tuple[float, float]]]:
        """Get shape points for a specific trip_id"""
        try:
            # Load trips to get shape_id
            trips_df = pd.read_csv(f"{self.data_dir}/trips.txt")
            trip_row = trips_df[trips_df['trip_id'] == trip_id]
            
            if trip_row.empty:
                logger.warning(f"No trip found for trip_id: {trip_id}")
                return None
            
            shape_id = trip_row.iloc[0]['shape_id']
            
            # Load shapes for this shape_id
            shapes_df = pd.read_csv(f"{self.data_dir}/shapes.txt")
            shape_points = shapes_df[shapes_df['shape_id'] == shape_id]
            
            if shape_points.empty:
                logger.warning(f"No shape points found for shape_id: {shape_id}")
                return None
            
            # Sort by sequence and convert to (lon, lat) tuples with regular Python floats
            shape_points = shape_points.sort_values('shape_pt_sequence')
            coords = [(float(lon), float(lat)) for lon, lat in zip(shape_points['shape_pt_lon'].values, shape_points['shape_pt_lat'].values)]
            
            logger.debug(f"Loaded {len(coords)} shape points for trip {trip_id}")
            # DEBUG: Show actual point count
            print(f"GTFS SHAPE DEBUG for {trip_id}: {len(coords)} points", file=sys.stderr)
            if len(coords) > 0:
                print(f"  First 3: {coords[:3]}", file=sys.stderr)
                print(f"  Last 3: {coords[-3:]}", file=sys.stderr)
            return coords
            
        except Exception as e:
            logger.error(f"Error getting shape for trip {trip_id}: {e}")
            return None

    def get_stop_sequence_for_trip(self, trip_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get the stop sequence for a specific trip_id with stop_times data"""
        try:
            # Load stop_times for this trip
            stop_times_df = pd.read_csv(f"{self.data_dir}/stop_times.txt")
            trip_stops = stop_times_df[stop_times_df['trip_id'] == trip_id]
            
            if trip_stops.empty:
                logger.warning(f"No stop_times found for trip_id: {trip_id}")
                return None
            
            # Sort by stop_sequence
            trip_stops = trip_stops.sort_values('stop_sequence')
            
            # Get stop information
            stops_df = pd.read_csv(f"{self.data_dir}/stops.txt")
            
            stop_sequence = []
            for _, row in trip_stops.iterrows():
                stop_id = row['stop_id']
                stop_info = stops_df[stops_df['stop_id'] == stop_id]
                
                if not stop_info.empty:
                    stop_sequence.append({
                        'stop_id': stop_id,
                        'stop_sequence': int(row['stop_sequence']),
                        'stop_name': str(stop_info.iloc[0]['stop_name']),
                        'stop_lat': float(stop_info.iloc[0]['stop_lat']),
                        'stop_lon': float(stop_info.iloc[0]['stop_lon'])
                    })
            
            logger.debug(f"Loaded {len(stop_sequence)} stops for trip {trip_id}")
            return stop_sequence
            
        except Exception as e:
            logger.error(f"Error getting stop sequence for trip {trip_id}: {e}")
            return None

    def slice_shape_between_stops(self, trip_id: str, from_stop_id: str, to_stop_id: str) -> Optional[List[Tuple[float, float]]]:
        """
        Slice the shape for a trip between two specific stops
        
        Args:
            trip_id: GTFS trip ID
            from_stop_id: Starting stop ID
            to_stop_id: Ending stop ID
            
        Returns:
            List of (lon, lat) coordinates for the sliced shape
        """
        try:
            # IMPROVED: Handle None or invalid stop IDs
            if not from_stop_id or not to_stop_id or from_stop_id == 'None' or to_stop_id == 'None':
                logger.warning(f"Invalid stop IDs for trip {trip_id}: from_stop_id={from_stop_id}, to_stop_id={to_stop_id}")
                return None
            
            # Get the full shape for this trip
            full_shape = self.get_shape_for_trip(trip_id)
            if not full_shape:
                logger.warning(f"No shape found for trip {trip_id}")
                return None
            
            # Get the stop sequence for this trip
            stop_sequence = self.get_stop_sequence_for_trip(trip_id)
            if not stop_sequence:
                logger.warning(f"No stop sequence found for trip {trip_id}")
                return None
            
            # Find the stops we want to slice between
            from_stop = None
            to_stop = None
            
            for stop in stop_sequence:
                if stop['stop_id'] == from_stop_id:
                    from_stop = stop
                if stop['stop_id'] == to_stop_id:
                    to_stop = stop
            
            if not from_stop or not to_stop:
                logger.warning(f"Stops {from_stop_id} or {to_stop_id} not found in trip {trip_id}")
                # IMPROVED: Try to find stops by name if ID lookup fails
                for stop in stop_sequence:
                    if stop.get('stop_name', '').lower() == str(from_stop_id).lower():
                        from_stop = stop
                    if stop.get('stop_name', '').lower() == str(to_stop_id).lower():
                        to_stop = stop
                
                # --- NEW FALLBACK -------------------------------------------------
                # If still missing, use coordinates from the global stops cache so
                # the shape can still be sliced by proximity even without the
                # stop appearing in this trip’s stop_times.txt.
                # ---------------------------------------------------------------
                if (not from_stop or not to_stop) and self.stops_cache:
                    pseudo_seq = 0
                    if not from_stop and str(from_stop_id) in self.stops_cache:
                        info = self.stops_cache[str(from_stop_id)]
                        from_stop = {
                            'stop_id': str(from_stop_id),
                            'stop_sequence': pseudo_seq,
                            'stop_name': info.get('name', str(from_stop_id)),
                            'stop_lat': info['lat'],
                            'stop_lon': info['lon'],
                        }
                        pseudo_seq += 1
                    if not to_stop and str(to_stop_id) in self.stops_cache:
                        info = self.stops_cache[str(to_stop_id)]
                        to_stop = {
                            'stop_id': str(to_stop_id),
                            'stop_sequence': pseudo_seq + 99,  # ensure >= from
                            'stop_name': info.get('name', str(to_stop_id)),
                            'stop_lat': info['lat'],
                            'stop_lon': info['lon'],
                        }

                if not from_stop or not to_stop:
                    logger.warning(f"Stops {from_stop_id} or {to_stop_id} still not found for trip {trip_id} – skipping GTFS shape slice.")
                    return None
            
            # Create a segment with just these two stops
            segment_stops = [from_stop, to_stop]
            
            # Now slice the shape between these stops
            segment_shape = self._slice_shape_by_stops(full_shape, segment_stops)
            
            if segment_shape:
                logger.debug(f"Sliced shape for trip {trip_id} from {from_stop_id} to {to_stop_id}: {len(segment_shape)} points")
                # DEBUG: Print the actual coordinates
                print(f"GTFS SHAPE DEBUG for {trip_id} ({from_stop_id} -> {to_stop_id}):", file=sys.stderr)
                print(f"  Points: {len(segment_shape)}", file=sys.stderr)
                if segment_shape:
                    print(f"  First 3: {segment_shape[:3]}", file=sys.stderr)
                    print(f"  Last 3: {segment_shape[-3:]}", file=sys.stderr)
                    # Check for coordinate format issues
                    for i, point in enumerate(segment_shape):
                        if not isinstance(point, tuple) or len(point) != 2:
                            print(f"  Invalid point {i}: {point}", file=sys.stderr)
                        elif not isinstance(point[0], (int, float)) or not isinstance(point[1], (int, float)):
                                                          print(f"  Non-numeric point {i}: {point} (types: {type(point[0])}, {type(point[1])})", file=sys.stderr)
                return segment_shape
            else:
                logger.warning(f"Failed to slice shape for trip {trip_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error slicing shape for trip {trip_id}: {e}")
            return None

    def _slice_shape_by_stops(self, full_shape: List[Tuple[float, float]], segment_stops: List[Dict[str, Any]]) -> Optional[List[Tuple[float, float]]]:
        """
        Slice the full shape to get only the portion between the specified stops
        Uses stop sequence order to ensure continuous paths
        Enforces min 25, preferred min 40 (with interpolation), max 60 (downsample)
        
        Args:
            full_shape: Full shape points as (lon, lat) tuples
            segment_stops: List of stops in the segment with coordinates
            
        Returns:
            Sliced shape points
        """
        if not full_shape or not segment_stops:
            return None
        
        try:
            # Sort stops by their sequence number to ensure proper order
            sorted_stops = sorted(segment_stops, key=lambda x: x['stop_sequence'])
            
            # Find the shape points closest to the first and last stops
            first_stop_coords = (sorted_stops[0]['stop_lat'], sorted_stops[0]['stop_lon'])  # (lat, lon)
            last_stop_coords = (sorted_stops[-1]['stop_lat'], sorted_stops[-1]['stop_lon'])  # (lat, lon)
            
            first_idx = self._find_closest_shape_point(full_shape, first_stop_coords[0], first_stop_coords[1])
            last_idx = self._find_closest_shape_point(full_shape, last_stop_coords[0], last_stop_coords[1])
            
            # ------------------------------------------------------------------
            # Choose the *shorter* of the two possible paths along the GTFS shape
            # (clock-wise vs counter-clock-wise) between the two stops.
            # This avoids figure-8 / back-tracking artefacts where the default
            # slice would follow the long way around the loop and confuse users.
            # ------------------------------------------------------------------

            def _path_length(seq: List[Tuple[float, float]]) -> float:
                """Approximate length (km) of a list of (lon, lat) points."""
                dist = 0.0
                for (lon1, lat1), (lon2, lat2) in zip(seq[:-1], seq[1:]):
                    dist += self._haversine_distance(lat1, lon1, lat2, lon2)
                return dist

            n_pts = len(full_shape)

            if first_idx <= last_idx:
                forward_slice = full_shape[first_idx:last_idx + 1]
                wrap_slice = full_shape[first_idx:] + full_shape[:last_idx + 1]
            else:
                # first_idx is after last_idx in the list
                forward_slice = full_shape[first_idx:] + full_shape[:last_idx + 1]
                # For the counter-clock-wise direction we need the slice in
                # *reverse* order so that it still starts from first_idx and
                # ends at last_idx.
                wrap_slice = list(reversed(full_shape[last_idx:first_idx + 1]))

            # Compute both lengths and pick the shorter path
            forward_len = _path_length(forward_slice)
            wrap_len = _path_length(wrap_slice)

            if forward_len <= wrap_len:
                sliced_shape = forward_slice
            else:
                sliced_shape = wrap_slice
            
            # ------------------------------------------------------------------
            # Trim potential duplicates: if the first/last point of the chosen
            # GTFS slice is practically the same as the stop coordinate we are
            # about to prepend/append, drop it.  This prevents micro-loops or
            # 1-pixel “boxes” at segment boundaries.
            # ------------------------------------------------------------------

            if sliced_shape:
                # Remove duplicating first point
                if self._haversine_distance(first_stop_coords[0], first_stop_coords[1],
                                             sliced_shape[0][1], sliced_shape[0][0]) < 0.00005:  # ≈5 m
                    sliced_shape = sliced_shape[1:]

                # Remove duplicating last point
                if sliced_shape and self._haversine_distance(last_stop_coords[0], last_stop_coords[1],
                                                             sliced_shape[-1][1], sliced_shape[-1][0]) < 0.00005:
                    sliced_shape = sliced_shape[:-1]

            # NOTE: sliced_shape is already in the correct direction because we
            # always start from first_idx and proceed until we hit last_idx by
            # concatenation rules above.

            # ------------------------------------------------------------------
            # Safety check: if the polyline we just built is absurdly longer
            # (> 3×) than the straight-line distance between the two stops,
            # replace it with a road-snapped path using the existing helper
            # (falls back to straight line when osmnx fails).  This removes
            # residual figure-8 artefacts on routes that bend back near their
            # origin later in the trip.
            # ------------------------------------------------------------------

            total_len_km = _path_length(sliced_shape)
            direct_len_km = self._haversine_distance(first_stop_coords[0], first_stop_coords[1],
                                                     last_stop_coords[0], last_stop_coords[1])

            if direct_len_km > 0 and total_len_km / direct_len_km > 3.0:
                logger.warning(
                    "GTFS shape between %s and %s is %.1fx longer than direct distance; "
                    "using road-snapped interpolation instead",
                    sorted_stops[0]['stop_id'], sorted_stops[-1]['stop_id'], total_len_km / direct_len_km)
                sliced_shape = self.interpolate_shape_between_stops(full_shape, first_stop_coords, last_stop_coords)
            
            # Add the actual stop coordinates at the beginning and end if they're not already included
            result_shape = []
            first_stop_result = (float(sorted_stops[0]['stop_lon']), float(sorted_stops[0]['stop_lat']))
            first_stop_for_distance = (float(sorted_stops[0]['stop_lat']), float(sorted_stops[0]['stop_lon']))
            # Always add the first stop as the first point
            result_shape.append(first_stop_result)
            
            # Preserve the original ordering of the GTFS shape slice so the
            # polyline follows the actual track. Re-ordering by radial distance
            # creates self-intersections/"figure-8" artefacts when the route
            # curves back toward the origin.
            result_shape.extend([(float(lon), float(lat)) for (lon, lat) in sliced_shape])
            
            last_stop_result = (float(sorted_stops[-1]['stop_lon']), float(sorted_stops[-1]['stop_lat']))
            last_stop_for_distance = (float(sorted_stops[-1]['stop_lat']), float(sorted_stops[-1]['stop_lon']))
            # Always add the last stop as the last point
            result_shape.append(last_stop_result)
            
            # DEBUG: Show what stops we're adding
            print(f"STOP ADDITION DEBUG:", file=sys.stderr)
            print(f"  Adding first stop: {first_stop_result}", file=sys.stderr)
            print(f"  Adding last stop: {last_stop_result}", file=sys.stderr)
            print(f"  Result shape after adding stops: {len(result_shape)} points", file=sys.stderr)
            if len(result_shape) > 0:
                print(f"  First point after adding stops: {result_shape[0]}", file=sys.stderr)
                print(f"  Last point after adding stops: {result_shape[-1]}", file=sys.stderr)
            
            # Enforce minimum and maximum with your requirements
            min_points = 4  # Minimum points per slice (for very short routes)
            preferred_min = 60  # Target 60+ points for most routes
            max_points = 200  # No strict upper limit, but reasonable max
            n = len(result_shape)
            
            # If too few points but at least min_points, interpolate up to preferred_min
            if n < preferred_min and n >= min_points:
                # Linear interpolation between points to reach 60+ points
                interp_points = []
                for i in range(n - 1):
                    interp_points.append(result_shape[i])
                    # Insert extra points between i and i+1 to reach preferred_min
                    num_interp = max(0, (preferred_min - n) // (n - 1))
                    for j in range(1, num_interp + 1):
                        frac = j / (num_interp + 1)
                        lon = result_shape[i][0] + frac * (result_shape[i+1][0] - result_shape[i][0])
                        lat = result_shape[i][1] + frac * (result_shape[i+1][1] - result_shape[i][1])
                        interp_points.append((lon, lat))
                interp_points.append(result_shape[-1])
                result_shape = interp_points
                n = len(result_shape)
            
            # If more than max_points, downsample to reasonable density
            if n > max_points:
                # Keep the stops and evenly spaced shape points
                middle_points = result_shape[1:-1]  # Exclude first and last
                if len(middle_points) > 0:
                    # Keep points spaced roughly 50-200m apart
                    target_middle_points = min(100, max_points - 2)  # Keep reasonable number
                    if target_middle_points > 0:
                        # Take evenly spaced points
                        step = len(middle_points) // target_middle_points
                        selected_middle = []
                        for i in range(0, len(middle_points), step):
                            if len(selected_middle) < target_middle_points:
                                selected_middle.append(middle_points[i])
                        # Ensure we keep the actual stops
                        first_stop = (float(sorted_stops[0]['stop_lon']), float(sorted_stops[0]['stop_lat']))
                        last_stop = (float(sorted_stops[-1]['stop_lon']), float(sorted_stops[-1]['stop_lat']))
                        result_shape = [first_stop] + selected_middle + [last_stop]
                    else:
                        # If we can only keep 2 points, just keep the stops
                        first_stop = (float(sorted_stops[0]['stop_lon']), float(sorted_stops[0]['stop_lat']))
                        last_stop = (float(sorted_stops[-1]['stop_lon']), float(sorted_stops[-1]['stop_lat']))
                        result_shape = [first_stop, last_stop]
                # If no middle points, result_shape is already correct (just 2 points)
            
            # DEBUG: Show the slicing details
            print(f"SHAPE SLICING DEBUG:", file=sys.stderr)
            print(f"  Full shape points: {len(full_shape)}", file=sys.stderr)
            print(f"  First stop: {sorted_stops[0]['stop_id']} ({sorted_stops[0]['stop_lat']:.6f}, {sorted_stops[0]['stop_lon']:.6f})", file=sys.stderr)
            print(f"  Last stop: {sorted_stops[-1]['stop_id']} ({sorted_stops[-1]['stop_lat']:.6f}, {sorted_stops[-1]['stop_lon']:.6f})", file=sys.stderr)
            print(f"  First stop -> shape point {first_idx}: distance = {self._distance(first_stop_coords, (full_shape[first_idx][1], full_shape[first_idx][0])):.6f}", file=sys.stderr)
            print(f"  Last stop -> shape point {last_idx}: distance = {self._distance(last_stop_coords, (full_shape[last_idx][1], full_shape[last_idx][0])):.6f}", file=sys.stderr)
            print(f"  Shape point {first_idx}: ({full_shape[first_idx][0]:.6f}, {full_shape[first_idx][1]:.6f})", file=sys.stderr)
            print(f"  Shape point {last_idx}: ({full_shape[last_idx][0]:.6f}, {full_shape[last_idx][1]:.6f})", file=sys.stderr)
            print(f"  Sliced points: {len(sliced_shape)}", file=sys.stderr)
            print(f"  Result points (after min/max): {len(result_shape)}", file=sys.stderr)
            if len(result_shape) > 0:
                print(f"  First 3: {result_shape[:3]}", file=sys.stderr)
                print(f"  Last 3: {result_shape[-3:]}", file=sys.stderr)
            return result_shape
        except Exception as e:
            logger.error(f"Error in _slice_shape_by_stops: {e}")
            return None
    
    def interpolate_shape_between_stops(self, 
                                      shape_points: List[Tuple[float, float]], 
                                      start_stop: Tuple[float, float], 
                                      end_stop: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Interpolate shape points between two stops using GTFS shape data
        
        Args:
            shape_points: Full shape points from GTFS
            start_stop: (lat, lon) of start stop
            end_stop: (lat, lon) of end stop
            
        Returns:
            List of interpolated (lon, lat) points
        """
        if not shape_points:
            return self._straight_line_interpolation(start_stop, end_stop)
        
        try:
            # Find the closest shape points to start and end stops
            start_idx = self._find_closest_shape_point(shape_points, start_stop[0], start_stop[1])
            end_idx = self._find_closest_shape_point(shape_points, end_stop[0], end_stop[1])
            
            # Ensure proper ordering
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            # Calculate the minimum number of points we want
            min_points = 4  # Minimum for very short routes
            preferred_min = 60  # Target 60+ points for most routes
            current_points = end_idx - start_idx + 1
            
            # If we have too few points, expand the range
            if current_points < preferred_min:
                # Calculate how many extra points we need
                needed_points = preferred_min - current_points
                buffer_points = max(10, needed_points // 2)  # At least 10 points on each side
                
                # Expand the range
                start_idx = max(0, start_idx - buffer_points)
                end_idx = min(len(shape_points) - 1, end_idx + buffer_points)
                
                # If we still don't have enough points, try to get more
                if (end_idx - start_idx + 1) < preferred_min:
                    # Try to get at least 60 points total
                    target_points = 60
                    available_points = len(shape_points)
                    if available_points >= target_points:
                        # Center the range around the stops
                        center = (start_idx + end_idx) // 2
                        half_range = target_points // 2
                        start_idx = max(0, center - half_range)
                        end_idx = min(len(shape_points) - 1, center + half_range)
            else:
                # Use a reasonable buffer for smoother polylines
                buffer_points = 10
                start_idx = max(0, start_idx - buffer_points)
                end_idx = min(len(shape_points) - 1, end_idx + buffer_points)
            
            # Extract the relevant portion of the shape
            relevant_points = shape_points[start_idx:end_idx + 1]
            
            # Just use the shape points - stops are already added in the main slicing logic
            interpolated_points = [(float(lon), float(lat)) for lon, lat in relevant_points]
            
            # DEBUG: Show interpolation details
            print(f"INTERPOLATION DEBUG: {len(interpolated_points)} points between stops", file=sys.stderr)
            print(f"  Shape points: {len(shape_points)}, sliced: {len(relevant_points)}", file=sys.stderr)
            print(f"  Range: {start_idx} to {end_idx}", file=sys.stderr)
            
            return interpolated_points
            
        except Exception as e:
            logger.error(f"Error interpolating shape: {e}")
            return self._straight_line_interpolation(start_stop, end_stop)
    
    def _find_closest_shape_point(self, shape_points: List[Tuple[float, float]], 
                                 target_lat: float, target_lon: float) -> int:
        """Find the index of the closest shape point to the target"""
        min_distance = float('inf')
        closest_idx = 0
        
        for i, (lon, lat) in enumerate(shape_points):
            distance = self._distance((target_lat, target_lon), (lat, lon))
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # DEBUG: Show what we found
        print(f"    Finding closest to ({target_lat:.6f}, {target_lon:.6f})", file=sys.stderr)
        print(f"    Found point {closest_idx}: ({shape_points[closest_idx][0]:.6f}, {shape_points[closest_idx][1]:.6f})", file=sys.stderr)
        print(f"    Distance: {min_distance:.6f}", file=sys.stderr)
        
        return closest_idx
    
    def _distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate approximate distance between two points in degrees"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    
    def _straight_line_interpolation(self, start_stop: Tuple[float, float], 
                                   end_stop: Tuple[float, float], 
                                   num_points: int = 200) -> List[Tuple[float, float]]:
        """
        Fallback to straight-line interpolation between stops
        
        Args:
            start_stop: (lat, lon) of start stop
            end_stop: (lat, lon) of end stop
            num_points: Number of interpolation points (increased to 200 for smoother polylines)
            
        Returns:
            List of interpolated (lon, lat) points
        """
        start_lat, start_lon = start_stop
        end_lat, end_lon = end_stop
        
        # Generate evenly spaced points
        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)
        
        # Convert to regular Python floats and (lon, lat) format for consistency
        result = [(float(lon), float(lat)) for lon, lat in zip(lons, lats)]
        
        # DEBUG: Show point count for straight line interpolation
        print(f"STRAIGHT LINE DEBUG: {len(result)} points from {start_stop} to {end_stop}", file=sys.stderr)
        if len(result) > 0:
            print(f"  First 3: {result[:3]}", file=sys.stderr)
            print(f"  Last 3: {result[-3:]}", file=sys.stderr)
        
        return result
    
    def _get_stop_coordinates(self, stop_id: str, segment: dict = None, endpoint: str = None) -> Optional[Tuple[float, float]]:
        """Get coordinates for a stop, returns (lat, lon). If stop_id is ORIGIN or DESTINATION, use segment data."""
        if stop_id in ("ORIGIN", "DESTINATION") and segment and endpoint:
            stop_info = segment.get(f'{endpoint}_stop', {})
            lat = stop_info.get('lat')
            lon = stop_info.get('lon')
            if lat is not None and lon is not None:
                return (float(lat), float(lon))
        stop = self.stops_cache.get(stop_id)
        if stop is None:
            return None
        # Handle both Stop objects and dicts
        if hasattr(stop, 'lat') and hasattr(stop, 'lon'):
            return (float(stop.lat), float(stop.lon))
        elif isinstance(stop, dict) and 'lat' in stop and 'lon' in stop:
            return (float(stop['lat']), float(stop['lon']))
        return None
    
    def generate_shapes_from_segments(self, segments: List[Dict[str, Any]]) -> List[List[Tuple[float, float]]]:
        """
        Generate shapes from route segments using GTFS shapes only:
        1. Use GTFS shapes for transit segments
        2. Use straight-line interpolation for walking segments
        
        Args:
            segments: List of route segments with trip_id and stop information
            
        Returns:
            List of shape point arrays
        """
        if not segments:
            logger.warning("No segments provided to generate_shapes_from_segments")
            return []
        
        shapes = []
        
        for segment in segments:
            try:
                trip_id = segment.get('trip_id')
                mode = segment.get('mode', 'Unknown')
                from_stop = segment.get('from')
                to_stop = segment.get('to')
                
                # Use improved coordinate lookup
                from_coords = self._get_stop_coordinates(from_stop, segment, 'from')
                to_coords = self._get_stop_coordinates(to_stop, segment, 'to')
                
                if not from_coords or not to_coords:
                    logger.warning(f"Could not get coordinates for segment: {from_stop} -> {to_stop}")
                    continue
                
                # Use GTFS shape for transit modes with proper slicing
                if trip_id and trip_id != 'WALKING' and mode in ['LRT', 'Bus', 'Jeep']:
                    # Use the new proper shape slicing method
                    sliced_shape = self.slice_shape_between_stops(trip_id, from_stop, to_stop)
                    if sliced_shape:
                        shapes.append(sliced_shape)
                        logger.debug(f"Generated GTFS_SLICED shape for {mode} trip {trip_id}")
                    else:
                        # If slicing failed, use straight line
                        straight_shape = self._straight_line_interpolation(from_coords, to_coords)
                        shapes.append(straight_shape)
                        logger.warning(f"Slicing failed for {mode} trip {trip_id}, using straight line")
                else:
                    # Use straight line for walking segments
                    straight_shape = self._straight_line_interpolation(from_coords, to_coords)
                    shapes.append(straight_shape)
                    logger.debug(f"Generated straight-line shape for {mode} segment")
                
            except Exception as e:
                logger.error(f"Exception in generate_shapes_from_segments for segment {segment}: {e}")
                # Add empty shape as fallback
                shapes.append([])
        
        return shapes
    
    def generate_route_shape(self, route_segments: List[Dict], 
                           stops_cache: Dict) -> List[Dict]:
        """
        Generate enhanced route shapes using unified approach
        
        Args:
            route_segments: List of route segments with trip_id and stop information
            stops_cache: Dictionary mapping stop_id to (lat, lon)
            
        Returns:
            List of route segments with enhanced shapes
        """
        enhanced_segments = []
        
        for segment in route_segments:
            enhanced_segment = segment.copy()
            
            # Use improved coordinate lookup
            start_stop_id = segment.get('from')
            end_stop_id = segment.get('to')
            start_stop = self._get_stop_coordinates(start_stop_id, segment, 'from')
            end_stop = self._get_stop_coordinates(end_stop_id, segment, 'to')
            
            if not start_stop or not end_stop:
                logger.warning(f"Stop coordinates not found for {start_stop_id} or {end_stop_id}")
                enhanced_segments.append(enhanced_segment)
                continue
            
            # Try to get GTFS shape for this trip
            trip_id = segment.get('trip_id')
            shape_points = None
            
            if trip_id:
                shape_points = self.get_shape_for_trip(trip_id)
            
            # Generate shape points using proper slicing
            if shape_points:
                # Use the new proper shape slicing method
                segment_shape = self.slice_shape_between_stops(trip_id, start_stop_id, end_stop_id)
                if segment_shape:
                    logger.debug(f"Generated GTFS_SLICED shape for segment {trip_id} with {len(segment_shape)} points")
                else:
                    # If slicing failed, use straight line
                    segment_shape = self._straight_line_interpolation(start_stop, end_stop)
                    logger.debug(f"Slicing failed for segment {trip_id}, using straight line with {len(segment_shape)} points")
            else:
                # Fallback to straight-line interpolation
                segment_shape = self._straight_line_interpolation(start_stop, end_stop)
                logger.debug(f"Generated straight-line shape for segment {trip_id} with {len(segment_shape)} points")
            
            # Convert to polyline format
            polyline = self._points_to_polyline(segment_shape)
            enhanced_segment['shape'] = polyline
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _points_to_polyline(self, points: List[Tuple[float, float]]) -> str:
        """
        Convert list of (lon, lat) points to polyline string
        
        Args:
            points: List of (lon, lat) tuples
            
        Returns:
            Polyline string
        """
        if not points:
            return ""
        
        # Simple polyline encoding
        polyline_parts = []
        for lon, lat in points:
            polyline_parts.append(f"{lon:.6f},{lat:.6f}")
        
        return ";".join(polyline_parts)
    
    def snap_point_to_shape(self, point: Tuple[float, float], polyline: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], int, float]:
        """
        Snap a point to the nearest location on a polyline
        
        Args:
            point: (lat, lon) tuple of the point to snap
            polyline: List of (lon, lat) tuples representing the polyline
            
        Returns:
            Tuple of (snapped_point, segment_index, distance)
        """
        if not polyline or len(polyline) < 2:
            return point, 0, 0.0
        
        min_distance = float('inf')
        closest_point = point
        closest_segment = 0
        
        # Check each segment of the polyline
        for i in range(len(polyline) - 1):
            start = polyline[i]
            end = polyline[i + 1]
            
            # Find closest point on this segment
            snapped, distance = self._closest_point_on_segment(point, start, end)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = snapped
                closest_segment = i
        
        return closest_point, closest_segment, min_distance
    
    def _closest_point_on_segment(self, point: Tuple[float, float], 
                                 start: Tuple[float, float], 
                                 end: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        """
        Find the closest point on a line segment to a given point
        
        Args:
            point: (lat, lon) tuple of the target point
            start: (lon, lat) tuple of segment start
            end: (lon, lat) tuple of segment end
            
        Returns:
            Tuple of (closest_point, distance)
        """
        point_lat, point_lon = point
        start_lon, start_lat = start
        end_lon, end_lat = end
        
        # Convert to radians for calculations
        lat1, lon1 = math.radians(start_lat), math.radians(start_lon)
        lat2, lon2 = math.radians(end_lat), math.radians(end_lon)
        lat_p, lon_p = math.radians(point_lat), math.radians(point_lon)
        
        # Calculate the closest point on the line
        if lat1 == lat2 and lon1 == lon2:
            # Segment is a point
            return (start_lat, start_lon), self._haversine_distance(point_lat, point_lon, start_lat, start_lon)
        
        # Calculate the fraction along the line
        t = ((lat_p - lat1) * (lat2 - lat1) + (lon_p - lon1) * (lon2 - lon1)) / \
            ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
        
        # Clamp to segment bounds
        t = max(0, min(1, t))
        
        # Calculate the closest point
        closest_lat = lat1 + t * (lat2 - lat1)
        closest_lon = lon1 + t * (lon2 - lon1)
        
        # Convert back to degrees
        closest_lat_deg = math.degrees(closest_lat)
        closest_lon_deg = math.degrees(closest_lon)
        
        # Calculate distance
        distance = self._haversine_distance(point_lat, point_lon, closest_lat_deg, closest_lon_deg)
        
        return (closest_lat_deg, closest_lon_deg), distance
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def get_shape_statistics(self) -> Dict:
        """Get statistics about available shapes"""
        if self.shapes_data is None:
            return {"total_shapes": 0, "total_points": 0, "total_trips": 0}
        
        total_shapes = len(self.shapes_cache)
        total_points = len(self.shapes_data)
        total_trips = len(self.trip_shapes)
        
        return {
            "total_shapes": total_shapes,
            "total_points": total_points,
            "total_trips": total_trips,
            "avg_points_per_shape": total_points / total_shapes if total_shapes > 0 else 0,
            "trips_with_shapes": len([t for t in self.trip_shapes.values() if t in self.shapes_cache])
        }
    
    def dump_stops_cache(self, filename='stops_cache_debug.pkl'):
        """Debug method to dump stops cache"""
        try:
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(self.stops_cache, f)
            logger.info(f"Dumped stops cache to {filename}")
        except Exception as e:
            logger.error(f"Failed to dump stops cache: {e}")
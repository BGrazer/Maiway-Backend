#!/usr/bin/env python3
"""
MaiWay Routing Engine - Flask Web API Blueprint
Multi-criteria routing system for commuter app
"""

from flask import Blueprint, request, jsonify
import json
import time
import math
import sys
from typing import Dict, Any, Optional, List
from routing.maiwayrouting.core_route_service import UnifiedRouteService
from routing.maiwayrouting.config import config
from routing.maiwayrouting.logger import logger
from routing.maiwayrouting.exceptions import MaiWayError, RouteNotFoundError, InvalidCoordinatesError
from routing.maiwayrouting.core_shape_generator import CoreShapeGenerator

routing_bp = Blueprint('routing_bp', __name__)

# Global route service instance
route_service: Optional[UnifiedRouteService] = None
unified_shape_generator = CoreShapeGenerator()

# Initialize route service and GTFS data ONCE at startup

def initialize_route_service():
    global route_service
    global unified_shape_generator
    try:
        config.validate()
        # Adjust data_dir to be relative to the project root
        config.data_dir = 'routing/data'
        route_service = UnifiedRouteService(config.data_dir)
        
        # Set up unified shape generator
        unified_shape_generator.set_stops_cache(route_service.stops)
        
        logger.info("Unified route service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize unified route service: {e}")
        raise

# Call this ONCE at startup
initialize_route_service()

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate coordinate bounds"""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def clean_nan_values(obj):
    """Recursively clean NaN values from objects to make them JSON serializable"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, (int, float)) and math.isinf(obj):
        return None
    else:
        return obj


@routing_bp.route('/routing/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if route_service is None:
            return jsonify({'status': 'error', 'message': 'Route service not initialized'}), 500
        
        return jsonify({
            'status': 'healthy',
            'message': 'MaiWay Routing Engine is running',
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@routing_bp.route('/routing', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'MaiWay Routing Engine',
        'version': '1.0.0',
        'description': 'Multi-criteria routing system for commuter app',
        'endpoints': {
            'health': '/routing/health',
            'route': '/routing/route',
            'search_stops': '/routing/search-stops'
        }
    })


@routing_bp.route('/routing/route', methods=['POST'])
def route():
    """Route endpoint (legacy, now uses multicriteria engine and returns 'fastest' route)"""
    try:
        data = request.get_json()
        print("[DEBUG] Incoming request data:", data)
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        start_coords = data.get('start', {})
        end_coords = data.get('end', {})
        preferences = data.get('preferences', ['fastest'])
        modes = data.get('modes', ['jeepney', 'bus', 'lrt', 'walking'])
        passenger_type = data.get('passenger_type', 'regular')
        if not start_coords or not end_coords:
            return jsonify({'error': 'Start and end coordinates required'}), 400
        start_lat = float(start_coords.get('lat', 0))
        start_lon = float(start_coords.get('lon', 0))
        end_lat = float(end_coords.get('lat', 0))
        end_lon = float(end_coords.get('lon', 0))
        if not validate_coordinates(start_lat, start_lon) or not validate_coordinates(end_lat, end_lon):
            return jsonify({'error': 'Invalid coordinates'}), 400
        # Use the new multicriteria routing API, but only return the first preference
        result = route_service.find_all_routes_with_coordinates(
            start_lat, start_lon, end_lat, end_lon,
            fare_type=passenger_type,
            preferences=preferences,
            allowed_modes=modes
        )
        # Robust error handling for None or error result
        if result is None or (isinstance(result, dict) and result.get('error')):
            error_msg = result.get('error') if isinstance(result, dict) and result.get('error') else 'No route found'
            logger.warning(f"/route: {error_msg}")
            # Return a 200 with empty route for frontend consistency
            key = data.get('mode', preferences[0])
            out = {key: [], "summary": {"fare_breakdown": {}, "total_cost": 0.0, "total_distance": 0.0}, "stops": []}
            return jsonify(out), 200
        response = format_multicriteria_response(result, preferences)
        response = clean_nan_values(response)
        # For legacy, just return the first preference's segments
        key = data.get('mode', preferences[0])
        # --- FIX: compute summary USING ONLY the segments that belong to the selected preference ---
        selected_segments = response.get(key, [])
        # Guard against missing/None
        if selected_segments is None:
            selected_segments = []

        summary = {
            "total_cost": sum(seg.get("fare", 0.0) for seg in selected_segments),
            "total_distance": sum(seg.get("distance", 0.0) for seg in selected_segments),
            "fare_breakdown": calculate_fare_breakdown(selected_segments),
        }

        # Build output using the recomputed summary
        out = {
            key: selected_segments,
            "summary": summary,
            # Limit stops to those in the selected route for clarity
            "stops": response.get("stops", []),
        }
        print(f"[DEBUG] API response for mode '{key}':", out)
        return jsonify(out)
    except Exception as e:
        logger.error(f"/route error: {e}")
        # Return a 500 with a clear error message
        return jsonify({'error': str(e)}), 500


@routing_bp.route('/routing/search-stops', methods=['GET'])
def search_stops():
    """Search for stops by name"""
    try:
        if route_service is None:
            return jsonify({'error': 'Route service not initialized'}), 500
        
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'suggestions': []})
        
        # Get stops from the graph builder
        stops = route_service.stops
        
        # Filter stops by name (case-insensitive)
        suggestions = []
        query_lower = query.lower()
        
        for stop in stops:
            stop_name = stop.get('name', '').lower()
            if query_lower in stop_name:
                suggestions.append({
                    'id': stop.get('id'),
                    'name': stop.get('name'),
                    'lat': stop.get('lat'),
                    'lon': stop.get('lon')
                })
                if len(suggestions) >= 10:  # Limit results
                    break
        
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        logger.error(f"Error in search stops: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@routing_bp.route('/routing/routes-multicriteria', methods=['POST'])
def routes_multicriteria():
    """Return fastest, convenient, and cheapest routes between coordinates, with real fares and summary."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        start_coords = data.get('start', {})
        end_coords = data.get('end', {})
        preferences = data.get('preferences', ['fastest', 'cheapest', 'convenient'])
        modes = data.get('modes', ['jeepney', 'bus', 'lrt', 'walking'])
        passenger_type = data.get('passenger_type', 'regular')
        if not start_coords or not end_coords:
            return jsonify({'error': 'Start and end coordinates required'}), 400
        start_lat = float(start_coords.get('lat', 0))
        start_lon = float(start_coords.get('lon', 0))
        end_lat = float(end_coords.get('lat', 0))
        end_lon = float(end_coords.get('lon', 0))
        if not validate_coordinates(start_lat, start_lon) or not validate_coordinates(end_lat, end_lon):
            return jsonify({'error': 'Invalid coordinates'}), 400
        # Call the routing engine with preferences, modes, and passenger_type
        result = route_service.find_all_routes_with_coordinates(
            start_lat, start_lon, end_lat, end_lon,
            fare_type=passenger_type,
            preferences=preferences,
            allowed_modes=modes
        )
        # Format response as required by frontend
        response = format_multicriteria_response(result, preferences)
        response = clean_nan_values(response)
        return jsonify(response)
    except Exception as e:
        logger.error(f"/routes-multicriteria error: {e}")
        return jsonify({'error': str(e)}), 500


def generate_instruction(segment: Dict[str, Any]) -> str:
    """Generate instruction text for a route segment (robust to string/dict)"""
    mode = segment.get('mode', 'Walking')
    from_stop = segment.get('from_stop', 'Unknown')
    to_stop = segment.get('to_stop', 'Unknown')
    print(f"[DEBUG] generate_instruction: from_stop={from_stop} to_stop={to_stop} mode={mode}")
    # Handle case where from_stop/to_stop might be strings or dictionaries
    if isinstance(from_stop, dict):
        from_stop_name = from_stop.get('name', from_stop.get('id', 'Unknown'))
    else:
        from_stop_name = str(from_stop)
    if isinstance(to_stop, dict):
        to_stop_name = to_stop.get('name', to_stop.get('id', 'Unknown'))
    else:
        to_stop_name = str(to_stop)
    if mode == 'Walking':
        if segment.get('reason') == 'first_mile':
            return f"Walk from origin to {to_stop_name}"
        elif segment.get('reason') == 'last_mile':
            return f"Walk from {from_stop_name} to destination"
        else:
            return f"Walk from {from_stop_name} to {to_stop_name}"
    else:
        route_id = segment.get('route_id', '')
        return f"Take {mode} from {from_stop_name} to {to_stop_name}"


def calculate_fare_breakdown(segments: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate fare breakdown by mode"""
    breakdown = {}
    
    for segment in segments:
        mode = segment.get('mode', 'Walking')
        fare = segment.get('fare', 0.0)
        
        if mode not in breakdown:
            breakdown[mode] = 0.0
        breakdown[mode] += fare
    
    return breakdown


def format_multicriteria_response(result, preferences):
    # Normalize mode names for frontend
    mode_map = {
        'Jeep': 'jeepney',
        'Bus': 'bus',
        'LRT': 'lrt',
        'Walking': 'walking',
        'Tricycle': 'tricycle',
        'walk': 'walking',
        'jeep': 'jeepney',
        'bus': 'bus',
        'lrt': 'lrt',
        'tricycle': 'tricycle',
    }
    out = {p: [] for p in preferences}
    all_stops = set()
    summary = {
        'total_cost': 0.0,
        'total_distance': 0.0,
        'fare_breakdown': {}
    }
    for pref in preferences:
        route = result.get(pref)
        if not route or not route.get('segments'):
            out[pref] = []
            continue
        segments = []
        for seg in route['segments']:
            mode = mode_map.get(seg.get('mode', '').lower().capitalize(), seg.get('mode', '').lower())
            from_stop = seg.get('from_stop', {})
            to_stop = seg.get('to_stop', {})

            # Robustly extract stop names and coordinates whether stop is a dict or plain string token (e.g. "ORIGIN")
            def _stop_info(stop_val, fallback_prefix):
                if isinstance(stop_val, dict):
                    name = stop_val.get('name', stop_val.get('id', fallback_prefix))
                    lat = stop_val.get('lat', seg.get(f'{fallback_prefix}_lat', 0.0))
                    lon = stop_val.get('lon', seg.get(f'{fallback_prefix}_lon', 0.0))
                else:
                    # Plain string such as 'ORIGIN' or 'DESTINATION'
                    name = str(stop_val)
                    lat = seg.get(f'{fallback_prefix}_lat', 0.0)
                    lon = seg.get(f'{fallback_prefix}_lon', 0.0)
                return name, lat, lon

            from_name, from_lat, from_lon = _stop_info(from_stop, 'from')
            to_name, to_lat, to_lon = _stop_info(to_stop, 'to')
            # Compose instruction and details
            instruction = seg.get('instruction') or generate_instruction(seg)
            detailed_instructions = seg.get('detailed_instructions', [instruction])
            name = seg.get('name') or seg.get('route_id') or mode.capitalize()
            segment_obj = {
                'mode': mode,
                'instruction': instruction,
                'name': name,
                'distance': seg.get('distance', 0.0),
                'fare': seg.get('fare', 0.0),
                'from_stop': {
                    'name': from_name,
                    'lat': from_lat,
                    'lon': from_lon
                },
                'to_stop': {
                    'name': to_name,
                    'lat': to_lat,
                    'lon': to_lon
                },
                'detailed_instructions': detailed_instructions
            }
            # Attach polyline directly to segment if available
            if seg.get('polyline'):
                segment_obj['polyline'] = seg['polyline']
            segments.append(segment_obj)
            all_stops.add((segment_obj['from_stop']['name'], segment_obj['from_stop']['lat'], segment_obj['from_stop']['lon']))
            all_stops.add((segment_obj['to_stop']['name'], segment_obj['to_stop']['lat'], segment_obj['to_stop']['lon']))
            # Fare breakdown
            if mode not in summary['fare_breakdown']:
                summary['fare_breakdown'][mode] = 0.0
            summary['fare_breakdown'][mode] += seg.get('fare', 0.0)
            summary['total_cost'] += seg.get('fare', 0.0)
            summary['total_distance'] += seg.get('distance', 0.0)
        out[pref] = segments
    out['summary'] = summary
    out['stops'] = [
        {'name': name, 'lat': lat, 'lon': lon}
        for (name, lat, lon) in all_stops
    ]
    return out
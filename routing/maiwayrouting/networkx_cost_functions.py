# Cost and heuristic functions for NetworkXRoutePlanner, matching src/routing/algorithms.py

TRANSFER_PENALTY_MIN = 120.0  # 2-hour perceived pain per transfer
FALLBACK_WALKING_PENALTY = 1500.0  # Extra pain for connectivity-fix walking edges
WALKING_PENALTY = 1000.0      # Discourage in-route walking unless unavoidable
FARE_DIVISOR = 10.0      # Fare has less influence unless cheapest preference

# Mode priorities for each route type (lower index = higher priority)
MODE_PRIORITIES = {
    'fastest':   ['lrt', 'jeepney', 'bus', 'walking'],
    'convenient':['bus', 'lrt', 'jeepney', 'walking'],
    'cheapest': ['jeepney', 'walking', 'bus', 'lrt'],
}

# Mode speeds (km/h) for time-based calculations
MODE_SPEEDS = {
    'lrt': 40.0,      # LRT is fastest
    'bus': 25.0,      # Bus is medium speed
    'jeepney': 20.0,  # Jeepney is slower due to frequent stops
    'jeep': 20.0,     # Alias for jeepney
    'walking': 5.0,   # Walking speed
    'walk': 5.0,      # Alias for walking
}

# Mode weights for different preferences (tuned)
MODE_WEIGHTS = {
    'fastest': {
        'lrt': 1.0,
        'bus': 1.5,
        'jeepney': 2.0,
        'jeep': 2.0,
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

MAX_FARE = 200  # Maximum fare constraint

def get_mode_id(edge):
    """Get mode from edge data, with proper normalization"""
    mode = edge.get('mode', '').lower()
    # Normalize mode names
    if mode in ['jeep', 'jeepney']:
        return 'jeepney'
    elif mode in ['walk', 'walking']:
        return 'walking'
    elif mode in ['lrt', 'light rail', 'light rail transit']:
        return 'lrt'
    elif mode in ['bus', 'coach']:
        return 'bus'
    else:
        return 'walking'  # Default fallback

def make_cost_function(mode_preference):
    """Create a cost function that properly weights LRT and uses real fares"""
    weights = MODE_WEIGHTS.get(mode_preference, MODE_WEIGHTS['fastest'])
    
    def cost(u, v, edge):
        mode = get_mode_id(edge)
        distance = edge.get('distance', 1.0)
        fare = edge.get('fare', 0.0)
        
        # Get mode weight
        mode_weight = weights.get(mode, weights.get('walking', 4.0))
        
        # Calculate base cost (distance * mode weight)
        base_cost = distance * mode_weight
        
        # Add fare cost (normalized)
        fare_cost = fare / FARE_DIVISOR  # Use new divisor
        
        # Special handling for LRT
        if mode == 'lrt':
            # LRT should be preferred for longer distances
            if distance > 6.0:  # For distances > 6km, LRT becomes more attractive
                base_cost *= 0.7  # Reduce LRT cost for longer distances
            # LRT should be preferred over multiple jeepney transfers
            if distance > 8.0:
                base_cost *= 0.6  # Even more attractive for longer distances
        
        # Penalize excessive fare
        if fare > MAX_FARE:
            base_cost += 10000
        
        # Add transfer penalty if this is a transfer edge
        if edge.get('type', '').lower() in ['transfer', 'walking_fallback']:
            base_cost += TRANSFER_PENALTY_MIN
        
        # STRONGLY penalize fallback walking edges
        if edge.get('type', '').lower() == 'walking_fallback':
            base_cost += FALLBACK_WALKING_PENALTY
        
        # Penalize regular walking more, and add distance-based penalty
        if mode in ['walk', 'walking']:
            base_cost += WALKING_PENALTY + (distance * 2.0)  # Add distance-based penalty
        
        # Add transfer penalty if mode changes
        # (not available here, but can be added if needed)
        
        return base_cost + fare_cost
    
    return cost

def transfer_aware_weight(u, v, data, mode_preference='fastest', prev_mode=None):
    """Transfer-aware weight function with proper LRT handling"""
    weight = data.get('weight', 1.0)
    mode = get_mode_id(data)
    
    # Get mode weight based on preference
    weights = MODE_WEIGHTS.get(mode_preference, MODE_WEIGHTS['fastest'])
    mode_weight = weights.get(mode, weights.get('walking', 4.0))
    
    # Apply mode priority penalty
    if mode_preference in MODE_PRIORITIES:
        priority = MODE_PRIORITIES[mode_preference].index(mode) if mode in MODE_PRIORITIES[mode_preference] else len(MODE_PRIORITIES[mode_preference])
        weight += priority * 0.5  # Each step down in priority adds 0.5 min
    
    # Special LRT handling
    if mode == 'lrt':
        distance = data.get('distance', 1.0)
        # LRT becomes more attractive for longer distances
        if distance > 6.0:
            weight *= 0.8  # Reduce weight for longer LRT trips
        if distance > 8.0:
            weight *= 0.7  # Even more attractive for very long trips
    
    # STRONGLY penalize fallback walking edges
    if data.get('type', '').lower() == 'walking_fallback':
        weight += FALLBACK_WALKING_PENALTY
    
    # Penalize regular walking more
    if mode in ['walk', 'walking']:
        weight += WALKING_PENALTY
    
    # Add transfer penalty if mode changes
    if prev_mode is not None and mode != prev_mode:
        weight += TRANSFER_PENALTY_MIN
    
    # For cheapest, add fare consideration
    if mode_preference == 'cheapest':
        fare = data.get('fare', 0.0)
        weight += fare / 5.0  # Normalize fare to minutes
    
    return weight

# Example: simple haversine heuristic (can be replaced with network-based if available)
def simple_heuristic(node, goal, node_coords):
    from math import radians, cos, sin, sqrt, atan2
    lat1, lon1 = node_coords[node]
    lat2, lon2 = node_coords[goal]
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c 
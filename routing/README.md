# MaiWay - Multi-Criteria Routing Engine

A Python-based routing engine for commuter apps using GTFS data, tricycle terminals, and fare tables. Features A* pathfinding with custom cost functions for different transport modes (LRT, Bus, Jeep, Tricycle, Walking) and accurate polylines using Mapbox.

## Features

- Multi-criteria routing: Fastest, cheapest, and most convenient routes
- A* pathfinding: Efficient route finding with custom cost functions
- GTFS integration: Full support for GTFS data (stops, routes, trips, shapes)
- Tricycle routing: Special handling for tricycle terminals with distance and highway restrictions
- Fare calculation: Accurate fare computation using distance-based and zone-based tables
- Mapbox integration: Accurate polylines for all transport modes
- Route consolidation: Merges consecutive segments for better user experience
- Transfer optimization: Minimizes unnecessary transfers with penalty system
- Walking constraints: Limits walking segments for realistic routes

## Quick Start

### Prerequisites

- Python 3.8+
- GTFS data files
- Mapbox API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gtfs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export MAPBOX_TOKEN="your_mapbox_api_key"
export DATA_DIR="data"
```

4. Run the application:
```bash
python routing.py
```

## Configuration

### Environment Variables

- MAPBOX_TOKEN: Mapbox API key for polyline generation
- DATA_DIR: Directory containing GTFS data files
- MAX_WALKING_DISTANCE: Maximum walking distance between stops (default: 0.3 km)
- MAX_TRICYCLE_DISTANCE: Maximum tricycle connection distance (default: 1.5 km)
- TRANSFER_PENALTY: Penalty for transfers (default: 10.0)
- MAX_WALKING_SEGMENTS: Maximum walking segments per route (default: 3)

### GTFS Data Structure

```
data/
├── agency.txt
├── routes.txt
├── stops.txt
├── trips.txt
├── stop_times.txt
├── shapes.txt
├── fares/
│   ├── lrt1_sj.csv
│   ├── lrt1_sv.csv
│   ├── pub_aircon.csv
│   ├── pub_ordinary.csv
│   └── puj.csv
└── tricycle.geojson
```

## API Endpoints

### Health Check
```
GET /health
```

### Route Finding
```
POST /route
{
    "start": {"lat": 14.5837, "lon": 120.9843},
    "end": {"lat": 14.5806, "lon": 120.9866},
    "mode": "fastest"
}
```

### Stop Search
```
GET /search-stops?q=station
```

## Architecture

### Core Components

- GraphBuilder: Builds routing graph from GTFS data
- AStarRouter: Implements A* pathfinding with multiple cost functions
- FareCalculator: Handles fare calculations for different modes
- ShapeGenerator: Generates polylines using Mapbox and GTFS shapes
- RouteConsolidator: Merges consecutive route segments
- TrikeConnector: Handles tricycle terminal connections

### Routing Modes

1. Fastest: Optimizes for travel time using mode-specific weights
2. Cheapest: Optimizes for fare cost with transfer penalties
3. Convenient: Balances time, cost, and transfer frequency

### Transport Modes

- LRT/MRT: Fixed rail transit with GTFS shapes
- Bus: Public bus service with Mapbox polylines
- Jeepney: Local jeepney service with Mapbox polylines
- Tricycle: Local tricycle service with distance restrictions
- Walking: Pedestrian connections with Mapbox polylines

## Polyline Generation

### Mapbox Integration
- Uses Mapbox Directions API for accurate road-following polylines
- Supports both driving and walking profiles
- Implements JSON file-based caching to reduce API calls
- Handles all transport modes except LRT/MRT

### GTFS Shapes
- LRT/MRT routes use GTFS shapes.txt for accurate rail alignment
- Provides precise station-to-station routing
- Maintains historical route accuracy

### Caching Strategy
- Polylines are cached in mapbox_polyline_cache.json
- Reduces API costs and improves response times
- Cache persists across application restarts

## Performance

- Graph building: ~10 seconds for Manila GTFS data
- Route finding: Sub-millisecond to tens of milliseconds
- Polyline generation: Cached responses are instant
- API response: Typically under 100ms for cached routes

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure
```
/
├── routing.py
├── requirements.txt
├── README.md
├── Dockerfile

├── data/
│   ├── fares/
│   ├── gtfs_manila/
│   ├── routes-geojson/
│   ├── [GTFS and geojson files...]
│   └── [backups, if needed]
├── cache/
│   └── [cache files]
├── scripts/
│   ├── filter_stops.py
│   ├── build_lrt_edges.py
│   ├── fix_lrt_distances.py
│   └── [other data-prep or utility scripts]
├── tests/
│   ├── test_multimodal_20.py
│   ├── test_lrt_multimodal.py
│   └── [other test files]
├── logs/
│   ├── maiway_20250713.log
│   ├── maiway_20250712.log
│   ├── maiway_20250709.log
│   ├── maiway_20250708.log
│   └── [other log files]
├── maiwayrouting/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── exceptions.py
│   ├── unified_route_service.py
│   ├── unified_shape_generator.py
│   ├── networkx_cost_functions.py
│   ├── models/
│   │   └── route_segments.py
│   ├── utils/
│   │   ├── fare_utils.py
│   │   └── geo_utils.py
│   ├── graph/
│   │   └── graph_builder.py
│   ├── routing/
│   │   └── algorithms.py
│   └── logging.json
```
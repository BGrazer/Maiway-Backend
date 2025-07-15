from dataclasses import dataclass
from typing import List, Optional, Set

@dataclass
class TransitSegment:
    """Represents a transit segment between consecutive stops"""
    from_stop: str
    to_stop: str
    trip_id: str
    route_id: str
    mode: str
    distance: float
    stop_sequence_from: int
    stop_sequence_to: int

@dataclass
class WalkingSegment:
    """Represents a walking segment"""
    from_stop: str
    to_stop: str
    distance: float
    reason: str  # "transfer", "first_mile", "last_mile"

@dataclass
class RouteSegment:
    """Unified route segment"""
    from_stop: str
    to_stop: str
    mode: str
    distance: float
    trip_id: Optional[str] = None
    route_id: Optional[str] = None
    reason: Optional[str] = None
    fare: float = 0.0  # Add fare field for fare-aware routing

@dataclass
class CompleteRoute:
    """Complete route with all segments"""
    segments: List[RouteSegment]
    total_distance: float
    transit_distance: float
    walking_distance: float
    num_transfers: int
    modes_used: Set[str] 
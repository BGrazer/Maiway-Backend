"""
Custom exceptions for MaiWay routing engine
"""

class MaiWayError(Exception):
    """Base exception for MaiWay routing engine"""
    pass


class GTFSDataError(MaiWayError):
    """Raised when there's an issue with GTFS data loading"""
    pass


class RouteNotFoundError(MaiWayError):
    """Raised when no route is found between origin and destination"""
    pass


class InvalidCoordinatesError(MaiWayError):
    """Raised when coordinates are invalid or out of bounds"""
    pass


class APIError(MaiWayError):
    """Raised when external API calls fail"""
    pass


class FareCalculationError(MaiWayError):
    """Raised when fare calculation fails"""
    pass


class GraphBuildError(MaiWayError):
    """Raised when graph building fails"""
    pass 
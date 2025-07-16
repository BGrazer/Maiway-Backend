"""
Configuration management for MaiWay routing engine
"""

import os
from typing import Optional


class Config:
    """Configuration class for MaiWay routing engine"""
    
    def __init__(self):
        # Data directories
        self.data_dir: str = os.getenv('DATA_DIR', 'routing_data')
        self.fares_dir: str = os.getenv('FARES_DIR', 'routing_data/fares')
        
        # Mapbox configuration
        self.mapbox_token: str = os.getenv('MAPBOX_TOKEN', 'pk.eyJ1IjoibWFpd2F5YWRtaW4iLCJhIjoiY21kM3IybmFvMDdrZTJscjZucXgxa2Q1byJ9.WUTTeidTzw-SYrhgtnlmMA')
        
        # Graph building parameters
        self.max_walking_distance: float = float(os.getenv('MAX_WALKING_DISTANCE', '0.3'))
        
        # Routing parameters
        self.transfer_penalty: float = float(os.getenv('TRANSFER_PENALTY', '10.0'))
        self.max_walking_segments: int = int(os.getenv('MAX_WALKING_SEGMENTS', '3'))
        
        # API configuration
        self.host: str = os.getenv('HOST', '0.0.0.0')
        self.port: int = int(os.getenv('PORT', '5000'))
        self.debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Logging
        self.log_level: str = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file: Optional[str] = os.getenv('LOG_FILE')
    
    def validate(self):
        """Validate configuration"""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        if not self.mapbox_token:
            raise ValueError("Mapbox token is required")
        
        if self.max_walking_distance <= 0:
            raise ValueError("Max walking distance must be positive")
    
    def get_graph_builder_config(self) -> dict:
        """Get configuration for GraphBuilder"""
        return {
            'data_dir': self.data_dir,
            'max_walking_distance': self.max_walking_distance
        }
    
    def get_router_config(self) -> dict:
        """Get configuration for AStarRouter"""
        return {
            'transfer_penalty': self.transfer_penalty,
            'max_walking_segments': self.max_walking_segments
        }
    
    def get_api_config(self) -> dict:
        """Get configuration for Flask API"""
        return {
            'host': self.host,
            'port': self.port,
            'debug': self.debug
        }


# Global configuration instance
config = Config() 
"""
Logging configuration for MaiWay routing engine
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import os


class MaiWayLogger:
    """Centralized logging for MaiWay routing engine"""
    
    def __init__(self, name: str = "maiway", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/maiway_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_route_request(self, origin: tuple, destination: tuple, mode: str, 
                         duration_ms: float, success: bool):
        """Log route request metrics"""
        self.info(f"Route request: {origin} -> {destination}, mode={mode}, "
                 f"duration={duration_ms:.2f}ms, success={success}")
    
    def log_api_call(self, api_name: str, duration_ms: float, success: bool):
        """Log API call metrics"""
        self.info(f"API call: {api_name}, duration={duration_ms:.2f}ms, success={success}")


# Global logger instance
logger = MaiWayLogger() 
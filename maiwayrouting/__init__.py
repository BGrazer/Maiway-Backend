__title__ = 'maiwayrouting'
__version__ = '1.0.0'
__author__ = 'MaiWay Team'
__contact__ = 'maiway@example.com'
__license__ = 'MIT'
__copyright__ = 'Copyright 2024 MaiWay Team'

__all__ = ['core_route_service', 'core_shape_generator', 'CoreShapeGenerator', 'networkx_cost_functions', 'config', 'logger', 'exceptions']

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())

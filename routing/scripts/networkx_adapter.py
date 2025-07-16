import networkx as nx
from .routingresult import RoutingResult

class NetworkXRoutePlanner:
    """
    Adapter for maiwayrouting to use a pre-built NetworkX graph for routing.
    Allows injection of custom graphs, weights, and cost functions.
    Bypasses the C backend and operates purely in Python.
    """
    def __init__(self, graph, cost_function=None, heuristic_function=None):
        """
        Args:
            graph: NetworkX DiGraph (multi-modal, with all relevant edge attributes)
            cost_function: function(u, v, data) -> float, returns edge weight for routing
            heuristic_function: function(node, goal) -> float, for A* (optional)
        """
        self.graph = graph
        self.cost_function = cost_function or (lambda u, v, data: data.get('weight', 1.0))
        self.heuristic_function = heuristic_function

    def find_path(self, source, target, mode='fastest'):
        """
        Find a path from source to target using the injected graph and cost function.
        Args:
            source: node id
            target: node id
            mode: user preference (e.g., 'fastest', 'cheapest', 'convenient')
        Returns:
            RoutingResult (compatible with maiwayrouting)
        """
        try:
            if self.heuristic_function:
                path = nx.astar_path(
                    self.graph, source, target,
                    weight=lambda u, v, d: self.cost_function(u, v, d, mode),
                    heuristic=lambda n, g: self.heuristic_function(n, g, mode)
                )
            else:
                path = nx.dijkstra_path(
                    self.graph, source, target,
                    weight=lambda u, v, d: self.cost_function(u, v, d, mode)
                )
        except nx.NetworkXNoPath:
            return RoutingResult(is_existent=False)
        except Exception as e:
            print(f"Routing error: {e}")
            return RoutingResult(is_existent=False)
        # Build RoutingResult (minimal, extend as needed)
        result = RoutingResult()
        result.is_existent = True
        result.path_by_vertices = path
        # Optionally, add more details (segments, cost, etc.)
        return result

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function

    def set_heuristic_function(self, heuristic_function):
        self.heuristic_function = heuristic_function 
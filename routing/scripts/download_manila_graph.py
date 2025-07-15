#!/usr/bin/env python3
"""
Download and cache Manila OSMnx graph for faster routing engine startup
"""

import osmnx as ox
import pickle
import os
import logging
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_manila_graph():
    """Download and cache the full Manila OSMnx graph"""
    
    cache_file = "manila_osmnx_graph.pkl"
    
    print("DOWNLOADING MANILA OSMNX GRAPH")
    print("=" * 50)
    
    # Check if cache already exists
    if os.path.exists(cache_file):
        print(f"Cache already exists: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                graph = pickle.load(f)
            print(f"Loaded cached graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    # Define Manila bounding box (correct format: south, west, north, east)
    manila_bbox = (14.4, 120.9, 14.8, 121.1)  # (south, west, north, east)
    
    print(f"Downloading Manila graph...")
    print(f"   Bounding box: {manila_bbox}")
    print(f"   This may take 5-10 minutes...")
    print(f"   Please wait...")
    
    try:
        # Show progress message
        print("Downloading road network data from OpenStreetMap...")
        
        # Download the graph using graph_from_place for better compatibility
        start_time = time.time()
        graph = ox.graph_from_place(
            "Manila, Philippines",
            network_type='drive',  # Use drive network for transit routing
            simplify=True
        )
        download_time = time.time() - start_time
        
        print(f"Downloaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        print(f"Download time: {download_time:.1f} seconds")
        
        # Cache the graph
        print(f"Caching graph to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(graph, f)
        
        print(f"Graph cached successfully!")
        print(f"Cache file: {os.path.abspath(cache_file)}")
        print(f"Size: {os.path.getsize(cache_file) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"Failed to download graph: {e}")
        return False

def verify_cache():
    """Verify the cached graph can be loaded"""
    
    cache_file = "manila_osmnx_graph.pkl"
    
    if not os.path.exists(cache_file):
        print("Cache file not found")
        return False
    
    try:
        with open(cache_file, 'rb') as f:
            graph = pickle.load(f)
        
        print(f"Cache verified: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return True
        
    except Exception as e:
        print(f"Failed to verify cache: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MANILA OSMNX GRAPH DOWNLOADER")
    print("=" * 50)
    
    # Download/cache the graph
    success = download_manila_graph()
    
    if success:
        print("\nSUCCESS!")
        print("The routing engine will now use the cached graph for faster startup.")
    else:
        print("\nFAILED!")
        print("The routing engine will fall back to basic routing without OSMnx.")
    
    print("\n" + "=" * 50) 
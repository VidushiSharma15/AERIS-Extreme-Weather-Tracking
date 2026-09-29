import sys
from pathlib import Path
import xarray as xr
import numpy as np

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from weather_core.gnn import WeatherGraphBuilder, SphericalMeshBuilder
from weather_core.preprocessing import WeatherPreprocessor

RAW_FILE = Path("D:/SIH26078_AERIS/data/raw/era5_amphan_2020.nc")


def visualize_spherical_graph():
    print("=" * 80)
    print("AERIS SPHERICAL GRAPH REPRESENTATION DIAGNOSTIC VISUALIZER")
    print("=" * 80)

    target_path = RAW_FILE if RAW_FILE.exists() else Path("D:/SIH26078_AERIS/data/processed/india_weather_sample_standardized.nc")
    if not target_path.exists():
        print(f"Error: Target dataset file not found at {target_path}")
        return

    # 1. Load weather dataset
    preprocessor = WeatherPreprocessor()
    ds_std, _ = preprocessor.preprocess_dataset(str(target_path), dataset_name="era5_amphan_2020")

    # 2. Build spherical graph topology
    builder = WeatherGraphBuilder(k_neighbors=4)
    graph_data = builder.build_graph_from_dataset(ds_std)

    num_nodes = graph_data["num_nodes"]
    num_edges = graph_data["num_edges"]
    num_features = graph_data["num_features"]
    grid_shape = graph_data["geo_metadata"]["grid_shape"]

    print(f"Dataset File Path   : {target_path}")
    print(f"Original 2D Grid    : {grid_shape[0]} lats x {grid_shape[1]} lons ({grid_shape[0] * grid_shape[1]} cells)")
    print(f"Graph Nodes (V)     : {num_nodes} spherical grid nodes")
    print(f"Graph Edges (E)     : {num_edges} geodesic neighborhood edges")
    print(f"Node Feature Dim (D): {num_features} dimensions per node")
    print(f"Edge Attribute Dim  : {graph_data['edge_attr'].shape[1]} dimensions (Geodesic Dist, Bearing)")

    # 3. Print 2D Grid vs Graph Representation Comparison Table
    print("\n" + "=" * 80)
    print("COMPARISON: ORIGINAL 2D GRID vs SPHERICAL GEODESIC GRAPH")
    print("=" * 80)
    print(f"{'Metric / Property':<30} | {'Original 2D Image Grid':<25} | {'Spherical Graph Topology':<25}")
    print("-" * 80)
    print(f"{'Spatial Geometry':<30} | {'Flat Euclidean 2D Grid':<25} | {'3D Geodesic Sphere (R=6371km)':<25}")
    print(f"{'Distance Calculation':<30} | {'Pixel dx, dy (Euclidean)':<25} | {'Haversine Geodesic (km)':<25}")
    print(f"{'Neighbor Relationships':<30} | {'4/8 Pixel Grid Neighbors':<25} | {'k-NN / Geodesic Radius':<25}")
    print(f"{'Polar Distortion':<30} | {'Severe at high latitudes':<25} | {'Uniform Spherical Geometry':<25}")
    print(f"{'Ensemble Support':<30} | {'Flat array stacks':<25} | {'Multi-layer graph nodes':<25}")

    # 4. Sample Node & Edge Topology Inspector
    print("\n" + "=" * 80)
    print("SAMPLE GRAPH NODE & EDGE INSPECTION")
    print("=" * 80)
    lats = graph_data["geo_metadata"]["latitudes"]
    lons = graph_data["geo_metadata"]["longitudes"]
    edge_idx = graph_data["edge_index"]
    edge_attr = graph_data["edge_attr"]

    for node_id in [0, num_nodes // 2, num_nodes - 1]:
        print(f"\nNode #{node_id:<5}: Location = ({lats[node_id]:.2f}°N, {lons[node_id]:.2f}°E)")
        print(f"  Node Feature Vector: {graph_data['x'][node_id].numpy().tolist()[:6]}...")

        # Find edges connected to node_id
        connected_edges = (edge_idx[0] == node_id).nonzero(as_tuple=True)[0]
        print(f"  Geodesic Edges Connected ({len(connected_edges)} neighbors):")
        for e in connected_edges[:3]:
            target_node = int(edge_idx[1][e])
            dist_km = float(edge_attr[e][0]) * 1000.0
            bearing = float(edge_attr[e][1]) * 360.0
            print(f"    -> Neighbor Node #{target_node:<5} ({lats[target_node]:.2f}°N, {lons[target_node]:.2f}°E) | "
                  f"Geodesic Dist = {dist_km:.1f} km | Bearing = {bearing:.1f}°")

    print("\n" + "=" * 80)
    print("GRAPH TOPOLOGY DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    visualize_spherical_graph()

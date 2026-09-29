import numpy as np
import torch
import xarray as xr
from typing import Dict, Any, List, Optional, Tuple
from .mesh import SphericalMeshBuilder


class WeatherGraphBuilder:
    """
    Constructs PyTorch-compatible spherical graph representations from real
    preprocessed weather datasets and anomaly metrics.
    """

    def __init__(self, k_neighbors: int = 4, radius_km: Optional[float] = None):
        self.k_neighbors = k_neighbors
        self.radius_km = radius_km
        self._cached_edge_index: Optional[torch.Tensor] = None
        self._cached_edge_attr: Optional[torch.Tensor] = None
        self._cached_dist_matrix: Optional[np.ndarray] = None
        self._cached_n_nodes: Optional[int] = None

    def build_graph_from_dataset(
        self,
        ds: xr.Dataset,
        anomaly_z_scores: Optional[np.ndarray] = None,
        ensemble_member: Optional[str] = None,
        forecast_lead_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Converts xarray Dataset grid into a spherical graph structure.
        """
        lats = ds.latitude.values
        lons = ds.longitude.values

        # Grid meshgrid flattening
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        flat_lats = lat_grid.flatten()
        flat_lons = lon_grid.flatten()
        n_nodes = len(flat_lats)

        # Extract available meteorological variables
        feature_list = []

        # 1. Normalized Latitude [-1, 1]
        feature_list.append(flat_lats / 90.0)
        # 2. Normalized Longitude [-1, 1]
        feature_list.append(flat_lons / 180.0)

        # 3. Weather variables if available
        var_names = ["10u", "10v", "msl", "tp", "t2m"]
        for var in var_names:
            if var in ds.data_vars:
                arr = ds[var].values
                # Flatten spatial grid (taking first timestep if multi-time)
                if arr.ndim == 3:
                    flat_var = arr[0].flatten()
                else:
                    flat_var = arr.flatten()

                # Normalize by standard scale
                std_val = np.std(flat_var) + 1e-6
                mean_val = np.mean(flat_var)
                norm_var = (flat_var - mean_val) / std_val
                feature_list.append(norm_var)

        # 4. Anomaly score feature
        if anomaly_z_scores is not None:
            flat_z = anomaly_z_scores.flatten()
            if len(flat_z) == n_nodes:
                feature_list.append(flat_z)
            else:
                feature_list.append(np.zeros(n_nodes, dtype=np.float32))
        else:
            feature_list.append(np.zeros(n_nodes, dtype=np.float32))

        # Stack into PyTorch FloatTensor [N, num_features]
        node_features_np = np.column_stack(feature_list).astype(np.float32)
        x_tensor = torch.tensor(node_features_np, dtype=torch.float32)

        # Check if edge graph is already cached for this node count
        if self._cached_n_nodes == n_nodes and self._cached_edge_index is not None:
            edge_index_tensor = self._cached_edge_index
            edge_attr_tensor = self._cached_edge_attr
        else:
            # Compute geodesic distance matrix
            dist_matrix = SphericalMeshBuilder.compute_geodesic_distance_matrix(flat_lats, flat_lons)

            src_nodes = []
            dst_nodes = []
            edge_dists = []
            edge_bearings = []

            flat_lat_rad = np.radians(flat_lats)
            flat_lon_rad = np.radians(flat_lons)

            for i in range(n_nodes):
                dists = dist_matrix[i]
                dists_copy = dists.copy()
                dists_copy[i] = np.inf

                if self.radius_km is not None:
                    neighbors = np.where((dists_copy > 0) & (dists_copy <= self.radius_km))[0]
                else:
                    neighbors = np.argsort(dists_copy)[: self.k_neighbors]

                for j in neighbors:
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    d_ij = dists[j]
                    edge_dists.append(d_ij)

                    y = np.sin(flat_lon_rad[j] - flat_lon_rad[i]) * np.cos(flat_lat_rad[j])
                    x_b = (
                        np.cos(flat_lat_rad[i]) * np.sin(flat_lat_rad[j])
                        - np.sin(flat_lat_rad[i]) * np.cos(flat_lat_rad[j]) * np.cos(flat_lon_rad[j] - flat_lon_rad[i])
                    )
                    bearing = (np.degrees(np.atan2(y, x_b)) + 360.0) % 360.0
                    edge_bearings.append(bearing)

            edge_index_tensor = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            edge_features_np = np.column_stack([np.array(edge_dists) / 1000.0, np.array(edge_bearings) / 360.0]).astype(np.float32)
            edge_attr_tensor = torch.tensor(edge_features_np, dtype=torch.float32)

            # Store in cache
            self._cached_n_nodes = n_nodes
            self._cached_edge_index = edge_index_tensor
            self._cached_edge_attr = edge_attr_tensor

        return {
            "x": x_tensor,
            "edge_index": edge_index_tensor,
            "edge_attr": edge_attr_tensor,
            "num_nodes": n_nodes,
            "num_edges": edge_index_tensor.shape[1],
            "num_features": x_tensor.shape[1],
            "geo_metadata": {
                "latitudes": flat_lats,
                "longitudes": flat_lons,
                "grid_shape": (len(lats), len(lons)),
                "ensemble_member": ensemble_member,
                "forecast_lead_time": forecast_lead_time,
            },
        }

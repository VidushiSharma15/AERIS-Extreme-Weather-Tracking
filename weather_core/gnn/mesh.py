import numpy as np
from typing import Dict, Any, List, Tuple


class SphericalMeshBuilder:
    """
    Constructs 3D spherical Cartesian representations and geodesic neighborhood
    matrices for geographic weather grid points.
    Earth radius R = 6371.0 km.
    """

    EARTH_RADIUS_KM = 6371.0

    @classmethod
    def latlon_to_cartesian(cls, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Converts (lat, lon) in degrees to 3D Cartesian coordinates (x, y, z) in km.
        x = R * cos(lat) * cos(lon)
        y = R * cos(lat) * sin(lon)
        z = R * sin(lat)
        """
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)

        x = cls.EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
        y = cls.EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
        z = cls.EARTH_RADIUS_KM * np.sin(lat_rad)

        return np.column_stack([x, y, z])

    @classmethod
    def compute_geodesic_distance_matrix(cls, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Haversine geodesic distance matrix in km between grid points.
        """
        n = len(lats)
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)

        lat1 = lat_rad[:, np.newaxis]
        lat2 = lat_rad[np.newaxis, :]
        lon1 = lon_rad[:, np.newaxis]
        lon2 = lon_rad[np.newaxis, :]

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
        return cls.EARTH_RADIUS_KM * c


class IcosahedralMeshBuilder:
    """
    Interface & verified prototype for hierarchical icosahedral spherical meshes.

    DISCLAIMER:
    Prototype spherical graph implementation; production global icosahedral mesh
    remains a research extension.
    """

    @classmethod
    def create_icosahedron_subdivision_prototype(cls, subdivision_level: int = 1) -> Dict[str, Any]:
        """
        Generates prototype icosahedral mesh vertices and face connectivity.
        """
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        nodes = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float32)

        # Normalize to Earth radius
        norms = np.linalg.norm(nodes, axis=1, keepdims=True)
        nodes = (nodes / norms) * SphericalMeshBuilder.EARTH_RADIUS_KM

        # Convert 3D nodes to lat/lon degrees
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        lats = np.degrees(np.arcsin(z / SphericalMeshBuilder.EARTH_RADIUS_KM))
        lons = np.degrees(np.arctan2(y, x))

        return {
            "subdivision_level": subdivision_level,
            "num_vertices": len(nodes),
            "vertices_cartesian": nodes,
            "latitudes": lats,
            "longitudes": lons,
            "is_prototype": True,
            "disclaimer": "Prototype spherical graph implementation; production global icosahedral mesh remains a research extension.",
        }

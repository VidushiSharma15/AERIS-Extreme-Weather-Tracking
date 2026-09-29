import unittest
import numpy as np
import pandas as pd
import torch
import xarray as xr
from weather_core.gnn import (
    SphericalMeshBuilder,
    IcosahedralMeshBuilder,
    WeatherGraphBuilder,
    SphericalGraphConv,
    WeatherGNNPredictor,
    GNNTrainer,
)


class TestGNNModule(unittest.TestCase):
    def setUp(self):
        # Create a small dummy weather Dataset (4 lats x 5 lons = 20 nodes)
        lats = np.array([10.0, 15.0, 20.0, 25.0])
        lons = np.array([80.0, 85.0, 90.0, 95.0, 100.0])
        times = pd.date_range("2026-09-25T00:00:00", periods=2, freq="6h")

        t2m = np.random.uniform(295, 305, size=(2, 4, 5)).astype(np.float32)
        msl = np.random.uniform(100000, 101300, size=(2, 4, 5)).astype(np.float32)

        self.ds = xr.Dataset(
            data_vars={
                "t2m": (["time", "latitude", "longitude"], t2m),
                "msl": (["time", "latitude", "longitude"], msl),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

    def test_spherical_mesh_builder(self):
        lats = np.array([0.0, 90.0])
        lons = np.array([0.0, 0.0])
        cart = SphericalMeshBuilder.latlon_to_cartesian(lats, lons)
        self.assertEqual(cart.shape, (2, 3))
        # Equator: (6371, 0, 0)
        self.assertAlmostEqual(cart[0, 0], 6371.0, places=1)
        # North Pole: (0, 0, 6371)
        self.assertAlmostEqual(cart[1, 2], 6371.0, places=1)

        dists = SphericalMeshBuilder.compute_geodesic_distance_matrix(lats, lons)
        self.assertEqual(dists.shape, (2, 2))

    def test_icosahedral_mesh_prototype(self):
        mesh = IcosahedralMeshBuilder.create_icosahedron_subdivision_prototype(subdivision_level=1)
        self.assertTrue(mesh["is_prototype"])
        self.assertEqual(mesh["num_vertices"], 12)
        self.assertIn("disclaimer", mesh)

    def test_weather_graph_builder(self):
        builder = WeatherGraphBuilder(k_neighbors=3)
        graph = builder.build_graph_from_dataset(self.ds)

        self.assertEqual(graph["num_nodes"], 20)
        self.assertEqual(graph["x"].shape[0], 20)
        self.assertEqual(graph["edge_index"].shape[0], 2)
        # Each node has 3 neighbors -> 20 * 3 = 60 edges
        self.assertEqual(graph["num_edges"], 60)

    def test_gnn_model_forward_pass(self):
        builder = WeatherGraphBuilder(k_neighbors=3)
        graph = builder.build_graph_from_dataset(self.ds)

        model = WeatherGNNPredictor(in_features=graph["num_features"], hidden_dim=16)
        model.eval()

        with torch.no_grad():
            node_scores, event_embed = model(graph["x"], graph["edge_index"])

        self.assertEqual(node_scores.shape, (20, 1))
        self.assertEqual(event_embed.shape, (1, 64))

    def test_gnn_trainer_step(self):
        builder = WeatherGraphBuilder(k_neighbors=3)
        graph = builder.build_graph_from_dataset(self.ds)

        trainer = GNNTrainer(in_features=graph["num_features"], learning_rate=0.001)
        loss = trainer.train_epoch(graph)
        self.assertIsInstance(loss, float)

        metrics = trainer.evaluate(graph)
        self.assertIn("loss", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)


if __name__ == "__main__":
    import pandas as pd
    unittest.main()

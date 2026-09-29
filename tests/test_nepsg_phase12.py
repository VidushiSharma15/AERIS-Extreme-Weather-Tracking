import unittest
import json
import torch
from pathlib import Path
from weather_core.ingestion.nwm_providers import NEPSGProvider
from weather_core.gnn.graph_builder import WeatherGraphBuilder
from weather_core.gnn.models import WeatherGNNPredictor
from weather_core.analysis.nepsg_phase12_pipeline import run_phase12_pipeline, PROCESSED_DIR, GRIB2_PATH


class TestNEPSGPhase12Pipeline(unittest.TestCase):
    def test_grib2_dataset_loading(self):
        self.assertTrue(GRIB2_PATH.exists(), f"Raw GRIB2 file missing at {GRIB2_PATH}")
        provider = NEPSGProvider()
        ds = provider.open_dataset(str(GRIB2_PATH))

        self.assertEqual(ds.attrs.get("data_source_type"), "NWP_FORECAST")
        self.assertEqual(ds.attrs.get("provider_wmo_code"), "dems")
        self.assertEqual(len(ds.lead_time), 13)
        self.assertEqual(len(ds.ensemble), 12)
        self.assertEqual(len(ds.latitude), 83)
        self.assertEqual(len(ds.longitude), 125)
        self.assertIn("total_precipitation", ds.data_vars)
        self.assertIn("10m_u_component_of_wind", ds.data_vars)
        self.assertIn("10m_v_component_of_wind", ds.data_vars)
        self.assertIn("mean_sea_level_pressure", ds.data_vars)

    def test_spherical_graph_and_gnn_forward_pass(self):
        provider = NEPSGProvider()
        ds = provider.open_dataset(str(GRIB2_PATH))
        ds_mean = ds.mean(dim="ensemble")

        builder = WeatherGraphBuilder(k_neighbors=4)
        graph = builder.build_graph_from_dataset(ds_mean.isel(lead_time=0), forecast_lead_time=0)

        self.assertEqual(graph["num_nodes"], 10375)
        self.assertEqual(graph["num_edges"], 41500)
        self.assertEqual(graph["x"].shape[0], 10375)
        self.assertEqual(graph["x"].shape[1], 3)

        model = WeatherGNNPredictor(in_features=graph["num_features"], hidden_dim=32)
        model.eval()
        with torch.no_grad():
            node_scores, event_embed = model(graph["x"], graph["edge_index"])

        self.assertEqual(node_scores.shape, (10375, 1))
        self.assertEqual(event_embed.shape, (1, 64))

    def test_phase12_full_pipeline_execution(self):
        summary = run_phase12_pipeline()

        self.assertEqual(summary["dataset_name"], "aeris_amphan_2020_multimodel_consensus")
        self.assertEqual(summary["data_source_type"], "HARMONIZED_MULTI_MODEL_CONSENSUS")
        self.assertEqual(summary["total_lead_steps"], 13)
        self.assertEqual(summary["contributing_independent_models"], ["NCMRWF", "ECMWF_IFS"])

        # Verify output files exist
        json_path = PROCESSED_DIR / "nepsg_amphan_events.json"
        events_geojson_path = PROCESSED_DIR / "nepsg_amphan_events.geojson"
        traj_geojson_path = PROCESSED_DIR / "nepsg_amphan_trajectories.geojson"
        embed_path = PROCESSED_DIR / "nepsg_amphan_gnn_embeddings.pt"

        self.assertTrue(json_path.exists())
        self.assertTrue(events_geojson_path.exists())
        self.assertTrue(traj_geojson_path.exists())
        self.assertTrue(embed_path.exists())

        # Check embedding file
        embed_data = torch.load(str(embed_path))
        self.assertEqual(embed_data["event_embeddings"].shape, (13, 64))

        # Check GeoJSON format
        with open(traj_geojson_path, "r") as f:
            traj_data = json.load(f)
        self.assertEqual(traj_data["type"], "FeatureCollection")
        self.assertGreater(len(traj_data["features"]), 0)


if __name__ == "__main__":
    unittest.main()

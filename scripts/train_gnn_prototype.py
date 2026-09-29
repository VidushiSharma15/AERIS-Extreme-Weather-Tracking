import sys
from pathlib import Path
import torch

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from weather_core.gnn import WeatherGraphBuilder, WeatherGNNPredictor, GNNTrainer
from weather_core.preprocessing import WeatherPreprocessor

RAW_FILE = Path("D:/SIH26078_AERIS/data/raw/era5_amphan_2020.nc")


def run_gnn_prototype_experiment():
    print("=" * 80)
    print("EXECUTING AERIS GNN PROTOTYPE FORWARD PASS & REPRODUCIBLE EXPERIMENT")
    print("=" * 80)

    target_path = RAW_FILE if RAW_FILE.exists() else Path("D:/SIH26078_AERIS/data/processed/india_weather_sample_standardized.nc")
    if not target_path.exists():
        print(f"Error: Target dataset file not found at {target_path}")
        return

    # 1. Load dataset & build graph
    preprocessor = WeatherPreprocessor()
    ds_std, _ = preprocessor.preprocess_dataset(str(target_path), dataset_name="era5_amphan_2020")

    builder = WeatherGraphBuilder(k_neighbors=4)
    graph_data = builder.build_graph_from_dataset(ds_std)

    # 2. Initialize GNN Predictor & Trainer
    trainer = GNNTrainer(learning_rate=0.002, checkpoint_dir="D:/SIH26078_AERIS/models")

    print(f"Compute Device Active : {trainer.device} ({torch.cuda.get_device_name(0) if trainer.device.type == 'cuda' else 'CPU Execution'})")
    print(f"Input Node Features   : {graph_data['x'].shape}")
    print(f"Input Edge Index      : {graph_data['edge_index'].shape}")

    # 3. Test GNN Forward Pass
    trainer.model.eval()
    x_dev = graph_data["x"].to(trainer.device)
    edge_dev = graph_data["edge_index"].to(trainer.device)

    with torch.no_grad():
        node_scores, event_embed = trainer.model(x_dev, edge_dev)

    print(f"\n--- GNN FORWARD PASS VERIFICATION ---")
    print(f"  Node Anomaly Scores Shape : {node_scores.shape}")
    print(f"  Event Embedding Vector    : {event_embed.shape}")
    print(f"  Sample Output Range       : Min = {node_scores.min().item():.4f}, Max = {node_scores.max().item():.4f}")

    # 4. Small Reproducible Training Experiment (5 Epochs)
    print(f"\n--- RUNNING REPRODUCIBLE TRAINING EXPERIMENT (5 EPOCHS) ---")
    initial_metrics = trainer.evaluate(graph_data)
    print(f"  [Epoch 0/5 (Initial)] -> Loss = {initial_metrics['loss']:.4f} | MAE = {initial_metrics['mae']:.4f} | RMSE = {initial_metrics['rmse']:.4f}")

    for epoch in range(1, 6):
        loss = trainer.train_epoch(graph_data)
        metrics = trainer.evaluate(graph_data)
        trainer.save_checkpoint(epoch=epoch, loss=loss)
        print(f"  [Epoch {epoch}/5] -> Loss = {metrics['loss']:.4f} | MAE = {metrics['mae']:.4f} | RMSE = {metrics['rmse']:.4f}")

    final_metrics = trainer.evaluate(graph_data)
    print("\n" + "=" * 80)
    print("GNN PROTOTYPE EXPERIMENT COMPLETE")
    print(f"  Final Loss : {final_metrics['loss']:.4f}")
    print(f"  Final MAE  : {final_metrics['mae']:.4f}")
    print(f"  Final RMSE : {final_metrics['rmse']:.4f}")
    print(f"  Checkpoint : Saved to D:\\SIH26078_AERIS\\models\\gnn_latest.pt & gnn_best.pt")
    print("=" * 80)


if __name__ == "__main__":
    run_gnn_prototype_experiment()

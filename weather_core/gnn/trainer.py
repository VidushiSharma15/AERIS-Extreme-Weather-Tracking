import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from .models import WeatherGNNPredictor


class GNNTrainer:
    """
    Training & evaluation pipeline manager for WeatherGNNPredictor.
    Supports CPU / CUDA device selection, checkpoint saving, and metric evaluation.
    """

    def __init__(
        self,
        model: Optional[WeatherGNNPredictor] = None,
        in_features: int = 8,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        checkpoint_dir: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model else WeatherGNNPredictor(in_features=in_features)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("D:/SIH26078_AERIS/models")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float("inf")

    def train_epoch(self, graph_data: Dict[str, Any]) -> float:
        self.model.train()
        x = graph_data["x"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)

        # Target: Node z-score anomaly feature (last column of x)
        target = x[:, -1:].to(self.device)

        self.optimizer.zero_grad()
        pred_scores, _ = self.model(x, edge_index)

        loss = self.criterion(pred_scores, target)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def evaluate(self, graph_data: Dict[str, Any]) -> Dict[str, float]:
        self.model.eval()
        x = graph_data["x"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        target = x[:, -1:].to(self.device)

        with torch.no_grad():
            pred_scores, _ = self.model(x, edge_index)
            loss = self.criterion(pred_scores, target).item()
            mae = float(torch.mean(torch.abs(pred_scores - target)).item())
            rmse = float(torch.sqrt(torch.mean((pred_scores - target) ** 2)).item())

        return {
            "loss": loss,
            "mae": mae,
            "rmse": rmse,
        }

    def save_checkpoint(self, is_best: bool = False, epoch: int = 0, loss: float = 0.0):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "device": str(self.device),
        }

        latest_path = self.checkpoint_dir / "gnn_latest.pt"
        torch.save(checkpoint, str(latest_path))

        if is_best or loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "gnn_best.pt"
            torch.save(checkpoint, str(best_path))

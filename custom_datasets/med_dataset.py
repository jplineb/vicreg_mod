from typing import Dict, List, Any, Optional
import torch
from torch import nn
from torchmetrics import AUROC
import numpy as np
from torchvision import transforms
from abc import ABC, abstractmethod


class MedicalDataset(ABC):
    """
    Abstract base class for medical imaging datasets that provides a common interface
    and shared functionality.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        gpu: Optional[int] = None,
        transforms_pytorch: str = "default",
    ):
        self.data_path = data_path
        self.gpu = gpu
        self.transforms = transforms_pytorch
        self.data_loader_spec = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )

        # Child classes should define these
        self.pathologies: List[str] = []
        self.pathologies_of_interest: List[str] = []

    @abstractmethod
    def get_dataset(self, split: str = "train") -> torch.utils.data.Dataset:
        """Get the underlying dataset for a specific split"""
        pass

    def get_dataloader(
        self, split: str = "train"
    ) -> None:
        """Return dataloader for a specific split"""
        raise NotImplementedError("Subclasses must implement this method")

    def check_dataloader(self, split: str = "train") -> None:
        """Utility method to inspect a batch from the dataloader"""
        print(next(iter(self.get_dataloader(split=split))))

    def calculate_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate BCE loss for multi-label classification"""
        targets = targets.nan_to_num(0)
        loss_func = nn.BCEWithLogitsLoss().cuda(self.gpu)
        loss = loss_func(predictions, targets.cuda(self.gpu, non_blocking=True).float())
        return loss

    def calculate_accuracy(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> np.ndarray:
        """Calculate binary accuracy for multi-label predictions"""
        outputs = torch.sigmoid(outputs).cpu()
        predicted = np.round(outputs)
        return predicted

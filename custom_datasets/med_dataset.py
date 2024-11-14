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
        self, split: str = "train", distributed: bool = False
    ) -> torch.utils.data.DataLoader:
        """Return dataloader for a specific split"""
        print(f"Fetching dataloader for {split} split")

        if distributed:
            if split != "train":
                raise NotImplementedError(
                    f"Distributed training not supported for {split} split"
                )

            dataset = self.get_dataset(split=split)
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = torch.utils.data.DataLoader(
                train_sampler, **self.data_loader_spec
            )
            return dataloader, train_sampler

        dataloader = torch.utils.data.DataLoader(
            self.get_dataset(split=split), **self.data_loader_spec
        )
        return dataloader

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

    def calculate_auc(
        self, outputs: List[torch.Tensor], targets: List[torch.Tensor]
    ) -> tuple:
        """Calculate various AUROC metrics"""
        num_classes = len(self.pathologies)

        # Calculate AUROC for all classes
        au_roc = AUROC(task="multilabel", num_labels=num_classes, average=None)
        au_roc_average = AUROC(task="multilabel", num_labels=num_classes)

        auc_calc_all = au_roc(torch.stack(outputs), torch.stack(targets).int())
        auc_roc_avg_all = au_roc_average(
            torch.stack(outputs), torch.stack(targets).int()
        )

        # Calculate per-pathology AUCs
        auc_dict = {}
        auc_of_interest = []
        for pathology, auc in zip(self.pathologies, auc_calc_all.tolist()):
            auc_dict[pathology] = auc
            if pathology in self.pathologies_of_interest:
                auc_of_interest.append(auc)

        # Calculate average AUC for pathologies of interest
        auc_of_avg_interest = np.mean(auc_of_interest) if auc_of_interest else 0

        return auc_calc_all, auc_roc_avg_all, auc_of_avg_interest, auc_dict

    def store_round_results(
        self,
        outputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        views: List[str],
        patient_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Store per-patient results including AUCs for each pathology"""
        au_roc = AUROC(
            task="multilabel", num_labels=len(self.pathologies), average=None
        )

        all_results = []
        for output, target, view, patient_id in zip(
            outputs, targets, views, patient_ids
        ):
            auc_calc = au_roc(output, target)

            results = {"patient_id": patient_id, "view": view}

            for pathology, auc in zip(self.pathologies, auc_calc):
                results[pathology] = auc

            all_results.append(results)

        return all_results

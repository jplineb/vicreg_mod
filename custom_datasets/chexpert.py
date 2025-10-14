import torchxrayvision as xrv
import torch
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

from torchmetrics import AUROC

from utils.log_config import configure_logging

logger = configure_logging()


class Chexpert:
    num_classes = 13
    multi_label = True
    """
    Chexpert paper: https://arxiv.org/pdf/1901.07031.pdf

    Labels:
        - 1,0,-1 have mentions
        - nan is not a mention

        Lets map the labels:
        -1: uncertain with mention -> {0,1} -> 0
        0: Negative mention -> 0
        1: Positive mention -> 1
        nan: no mention -> 0
    """

    path_to_chexpert = (
        "/project/dane2/wficai/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/"
    )

    def __init__(
        self,
        percent=100,
        transforms_pytorch="RGB",
        batch_size=64,
        num_workers=0,
        gpu=None,
        train_views=["PA", "AP"],
        valid_views=["PA", "AP"],
    ):
        self.percent = percent
        self.transforms = transforms_pytorch
        self.data_loader_spec = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        self.train_views = train_views
        self.valid_views = valid_views
        self.gpu = gpu
        self.pathologies_of_interest = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
        ]

        logger.info(f"using views: {self.train_views}, {self.valid_views}")

    @property
    def transforms_pytorch(self):
        if self.transforms == "default":
            tfsms = transforms.Compose(
                [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
            )
        elif self.transforms == "RGB":
            tfsms = transforms.Compose(
                [
                    xrv.datasets.XRayCenterCrop(),
                    xrv.datasets.XRayResizer(224),
                    xrv.datasets.ToPILImage(),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                ]
            )
        else:
            tfsms = self.transforms

        return tfsms

    def get_dataset(self, split="train") -> torch.utils.data.Dataset:
        if split == "train":
            d_chex = xrv.datasets.CheX_Dataset(
                imgpath=self.path_to_chexpert,
                csvpath=f"{self.path_to_chexpert}/train.csv",
                transform=self.transforms_pytorch,
                views=self.train_views,
                unique_patients=False,
            )
        if split == "valid":
            d_chex = xrv.datasets.CheX_Dataset(
                imgpath=self.path_to_chexpert,
                csvpath=f"{self.path_to_chexpert}/valid.csv",
                transform=self.transforms_pytorch,
                views=self.valid_views,
                unique_patients=False,
            )

        if split == "test":
            raise NotImplementedError

        self.pathologies = d_chex.pathologies

        return d_chex

    def check_dataset(self, idx=1, split="train") -> dict:
        # Get idx
        instance = self.get_dataset(split=split)[idx]
        # Print labels
        print(f"labels: {instance['lab']}")
        try:
            # Show image
            plt.imshow(instance["img"].squeeze())
        except:
            # Transpose dimensions if necessary
            plt.imshow(instance["img"].transpose(2, 0))

        return instance

    def get_dataloader(
        self, split="train", distributed=False
    ) -> torch.utils.data.DataLoader:
        """
        Return dataloader given a certain split
        """
        logger.info(f"Fetching dataloader for {split} split")
        if distributed:
            if split != "train":
                logger.info(f"Probably shouldn't use {split} for distrbuted training")
                raise NotImplementedError
            else:
                dataset = self.get_dataset(split=split)
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                dataloader = torch.utils.data.DataLoader(
                    train_sampler, **self.data_loader_spec
                )
                return dataloader, train_sampler
        else:
            dataloader = torch.utils.data.DataLoader(
                self.get_dataset(split=split), **self.data_loader_spec
            )

        return dataloader

    def check_dataloader(self, split="train"):
        print(next(iter(self.get_dataloader(split=split))))

    def calculate_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        targets = targets.nan_to_num(0)
        loss_func = nn.BCEWithLogitsLoss().cuda(self.gpu)
        loss = loss_func(predictions, targets.cuda(self.gpu, non_blocking=True))
        return loss

    def calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        https://discuss.pytorch.org/t/how-to-calculate-accuracy-for-multi-label-classification/94906/2
        NOTE: Had to implement in code (ie. not used right now)
        """
        outputs = torch.sigmoid(outputs).cpu()
        predicted = np.round(outputs)
        return predicted

    def calculate_auc(self, outputs, targets)->tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict]:
        """
        Calculate AUROC metrics for multilabel classification:
        - Per-label AUROC (average=None)
        - Macro average AUROC (average="macro")
        - Weighted average AUROC (average="weighted")
        """
        outputs = torch.stack(outputs)
        targets = torch.stack(targets).int()

        # Per-label AUROC
        au_roc_none = AUROC(task="multilabel", num_labels=14, average=None)
        auc_calc_all = au_roc_none(outputs, targets)

        # Macro average AUROC
        au_roc_macro = AUROC(task="multilabel", num_labels=14, average="macro")
        auc_calc_macro = au_roc_macro(outputs, targets)

        # Weighted average AUROC
        au_roc_weighted = AUROC(task="multilabel", num_labels=14, average="weighted")
        auc_calc_weighted = au_roc_weighted(outputs, targets)

        # Get the AUROCs of interest
        auc_dict = {}
        auc_of_interest = []
        for pathology, auc in zip(self.pathologies, auc_calc_all.tolist()):
            auc_dict[pathology] = auc
            if pathology in self.pathologies_of_interest:
                auc_of_interest.append(auc)
        auc_of_avg_interest = np.mean(auc_of_interest)  

        return auc_calc_all, auc_calc_macro, auc_calc_weighted, auc_of_avg_interest, auc_dict

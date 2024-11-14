import torchxrayvision as xrv
import torch
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

from torchmetrics import AUROC


class Chexpert:
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

    path_to_chexpert = "/project/dane2/wficai/chexpert/chexpertchestxrays-u20210408/"

    def __init__(
        self,
        percent=100,
        transforms_pytorch="default",
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

        print(f"using views: {self.train_views}, {self.valid_views}")

    @property
    def transforms_pytorch(self):
        if self.transforms == "default":
            tfsms = transforms.Compose(
                [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
            )
        if self.transforms == "RGB":
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
        print(f"Fetching dataloader for {split} split")
        if distributed:
            if split != "train":
                print(f"Probably shouldn't use {split} for distrbuted training")
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

    def calculate_auc(self, outputs, targets):
        # Calculate metrics
        """
        NOTE: setting average = None allows us to see score for every label
        https://torchmetrics.readthedocs.io/en/stable/classification/auroc.html
        """
        # Calculate AUROC for the whole stack
        au_roc = AUROC(task="multilabel", num_labels=13, average=None)
        au_roc_average = AUROC(task="multilabel", num_labels=13)
        auc_calc_all = au_roc(torch.stack(outputs), torch.stack(targets).int())
        auc_roc_avg_all = au_roc_average(
            torch.stack(outputs), torch.stack(targets).int()
        )

        # Get the AUROCs of interest
        auc_dict = {}
        auc_of_interest = []
        for pathology, auc in zip(self.pathologies, auc_calc_all.tolist()):
            # Store in master dict
            auc_dict[pathology] = auc
            # If AUC in pathologies of interest put them in list
            if pathology in self.pathologies_of_interest:
                auc_of_interest.append(auc)

        # Get average of AUROC of interest
        auc_of_avg_interest = np.mean(auc_of_interest)

        return auc_calc_all, auc_roc_avg_all, auc_of_avg_interest, auc_dict

    def store_round_results(self, outputs, targets, views, patient_ids):
        au_roc = AUROC(task="multilabel", num_labels=13, average=None)

        all_results = []
        for output, target, view, patient_id in zip(
            outputs, targets, views, patient_ids
        ):

            auc_calc = au_roc(output, target)
            # Create a dict for each row
            results = dict(patient_id=patient_id, view=view)
            # Loop and store in results
            for pathology, auc in zip(self.pathologies, auc_calc):
                results[pathology] = auc

            all_results.append(results)

        return all_results

from matplotlib import pyplot as plt
from torchmetrics import AUROC
from torchvision import transforms
from torchvision.datasets import VisionDataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F


class BCN20000Base(VisionDataset):
    """
    ISIC2019 Dataset has a goal of classifying dermoscopic images among nine different diagnostic categories. 25,331 images are available for training across 8 different categories. We transformed them into 5 classes.

    After download, put your files under a folder called isic2019, then under a folder called dermatology under your data root.
    """

    # mean: tensor([0.6678, 0.5298, 0.5245])
    # std:  tensor([0.2231, 0.2029, 0.2145])
    input_size = (224, 224)
    patch_size = (16, 16)
    in_channels = 3
    num_classes = 5
    classes = ["MEL", "NV", "BCC", "AKIEC", "OTHER"]
    label_fracs = {"small": 8, "medium": 64, "large": 256, "full": np.inf}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms_pytorch: str | None | transforms.Compose = "default",
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self._transforms_pytorch = transforms_pytorch
        assert self._transforms_pytorch is not None
        super().__init__(root, transform=self.transforms_pytorch)

        self.index_file = pd.read_csv(
            os.path.join(self.root, f"{self.split}_index.csv")
        )

    @property
    def transforms_pytorch(self) -> transforms.Compose:
        if self._transforms_pytorch == "default":
            return transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(
                        self.input_size[0] - 1, max_size=self.input_size[0]
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.6678, 0.5298, 0.5245], [0.2231, 0.2029, 0.2145]
                    ),
                ]
            )
        elif self._transforms_pytorch is None:
            raise ValueError("Must pass transforms")

    def __len__(self) -> int:
        return len(self.index_file)

    def __getitem__(self, index: int) -> dict:
        df_row = self.index_file.iloc[index]
        fname = df_row["image_name"]
        label = df_row[self.classes].values.astype(np.int64)
        # label = df_row[self.classes].values

        img = Image.open(fname)
        if self.transforms_pytorch is not None:
            img = self.transforms_pytorch(img)

        sample = {}
        sample["idx"] = index

        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            # edge case 223,223,  resize to match 224*224
            dim_gap = self.input_size[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)

        sample["lab"] = label
        sample["img"] = img
        sample["fname"] = fname

        return sample


class BCN20000:
    """ISIC2019 (BCN_20000)Dataset has a goal of classifying dermoscopic images among nine different diagnostic categories. 25,331 images are available for training across 8 different categories. We transformed them into 5 classes."""

    input_size = (224, 224)
    patch_size = (16, 16)
    in_channels = 3
    num_classes = 5
    classes = ["MEL", "NV", "BCC", "AKIEC", "OTHER"]

    def __init__(
        self,
        path_to_bcn_20000="/project/dane2/wficai/BenchMD/isic2019/",
        transforms_pytorch="default",
        batch_size=64,
        num_workers=0,
        gpu=None,
        train_split_frac: float = 0.8,
    ):
        self.path_to_bcn_20000 = path_to_bcn_20000
        self.transforms = transforms_pytorch
        self.gpu = gpu
        self.train_split_frac = train_split_frac
        self.root = path_to_bcn_20000
        self.num_workers = num_workers
        self.batch_size = batch_size

    def build_index(self) -> None:
        print("Building index...")
        index_file = os.path.join(
            self.path_to_bcn_20000, "ISIC_2019_Training_GroundTruth.csv"
        )
        df = pd.read_csv(index_file)
        df["image_name"] = df["image"].apply(
            lambda s: os.path.join(self.root, "ISIC_2019_Training_Input/" + s + ".jpg")
        )

        # merge bkl, df, vasc into other, since they are less frequent classes
        df["OTHER"] = np.where(
            (df["BKL"] == 1) | (df["DF"] == 1) | (df["VASC"] == 1), 1, 0
        )
        # merge ack and scc into one class akiec
        df["AKIEC"] = np.where((df["AK"] == 1) | (df["SCC"] == 1), 1, 0)
        df = df.drop(columns=["BKL", "DF", "VASC", "UNK", "AK", "SCC"])

        # Clean up negative values
        for c in self.classes:
            df.loc[(df[c] < 0), c] = 0

        df["labels"] = df[self.classes].idxmax(axis=1)

        # Create train and validation splits
        train_index = pd.DataFrame(columns=df.columns)
        for label in df["labels"].unique():
            df_sub = df[df["labels"] == label]
            train_subset = df_sub.sample(frac=self.train_split_frac, replace=False)
            train_index = pd.concat([train_index, train_subset])

        # Get validation set as remaining samples
        valid_index = pd.concat([df, train_index]).drop_duplicates(keep=False)

        # Save splits to files
        train_output = os.path.join(self.path_to_bcn_20000, "train_index.csv")
        valid_output = os.path.join(self.path_to_bcn_20000, "valid_index.csv")

        train_index.to_csv(train_output, index=False)
        print(f"Train split saved to {train_output}")
        valid_index.to_csv(valid_output, index=False)
        print(f"Validation split saved to {valid_output}")
        print("Done")

    def get_dataset(self, split="train"):
        if split == "train":
            d_bcn_20000 = BCN20000Base(
                self.root, split="train", transforms_pytorch=self.transforms
            )
        elif split == "valid":
            d_bcn_20000 = BCN20000Base(
                self.root, split="valid", transforms_pytorch=self.transforms
            )
        else:
            raise ValueError(f"Invalid split: {split}")
        print(f"length of dataset {split}: {len(d_bcn_20000)}")
        return d_bcn_20000

    def get_dataloader(self, split: str = "train") -> torch.utils.data.DataLoader:
        d_bcn_20000 = self.get_dataset(split)
        dataloader = torch.utils.data.DataLoader(
            d_bcn_20000,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def check_dataset(self, idx: int = 1, split: str = "train") -> None:
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

    def check_dataloader(self, split: str = "train") -> None:
        print(next(iter(self.get_dataloader(split))))

    def calculate_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # Convert one-hot encoded targets to class indices
        targets = torch.argmax(targets, dim=1)
        # Calculate loss
        loss_func = torch.nn.CrossEntropyLoss().cuda(self.gpu)
        loss = loss_func(predictions, targets.cuda(self.gpu, non_blocking=True))
        return loss

    def calculate_auc(self, outputs, targets) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate AUROC
        outputs = torch.stack(outputs)
        targets = torch.stack(targets)

        # Convert one-hot encoded targets to class indices
        targets = torch.argmax(targets, dim=1).long()  # Now shape = [N]

        # Apply softmax to outputs
        # NOTE: this is not needed, since the loss function is already a softmax
        outputs = F.softmax(outputs, dim=1)

        au_roc = AUROC(task="multiclass", num_classes=self.num_classes, average=None)
        au_roc_average = AUROC(task="multiclass", num_classes=self.num_classes)

        # import pdb; pdb.set_trace()

        auc_calc_all = au_roc(outputs, targets)
        auc_calc_avg = au_roc_average(outputs, targets)

        return auc_calc_all, auc_calc_avg

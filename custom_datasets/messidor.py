from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchmetrics import AUROC
from sklearn.model_selection import StratifiedKFold


class MessidorBase(VisionDataset):
    """
    A dataset class for the Messidor 2 dataset, ophthalmology dataset, grading diabetic retinopathy on the 0-4 Davis Scale.
    (https://www.adcis.net/en/third-party/messidor2/)
    (https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)
    Note that you must register and manually download the data to use this dataset. Download the main folder from the adcis.net
    link, and extract all the files to a "messidor2" folder in this directory. It should contain messidor-2.csv and an "IMAGES" directory.
    Then add the messidor_data.csv and messidor_readme.txt files from the kaggle link to the "messidor2" directory as well.
    """

    num_classes = 5
    input_size = (224, 224)
    patch_size = (16, 16)
    in_channels = 3

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms_pytorch: str | None | transforms.Compose = "default",
        index_file_path: str | None = None,
    ):
        self._transforms_pytorch = transforms_pytorch
        assert self._transforms_pytorch is not None
        super().__init__(root, transform=self.transforms_pytorch)

        self.split = split
        if index_file_path is not None:
            self.index_file = pd.read_csv(index_file_path)
        else:
            self.index_file = pd.read_csv(
                os.path.join(self.root, f"{self.split}_index.csv")
            )

    @property
    def transforms_pytorch(self) -> transforms.Compose:
        """
        
        """
        if self._transforms_pytorch == "default":
            return transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(
                        self.input_size[0] - 1, max_size=self.input_size[0]
                    ),  # resizes (H,W) to (149, 224)
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]
                    ),
                    transforms.Pad((0, 37, 0, 38)),
                ]
            )
        if self._transforms_pytorch == "no_normalize":
            return transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(
                        self.input_size[0] - 1, max_size=self.input_size[0]
                    ),
                    transforms.ToTensor(),
                    transforms.Pad((0, 37, 0, 38)),
                ]
            )
        elif self._transforms_pytorch is None:
            raise ValueError("Must pass transforms")

    def __getitem__(self, index):
        df_row = self.index_file.iloc[index]
        fname = df_row["image_id"]
        label = df_row["adjudicated_dr_grade"]

        # Load image
        img_path = os.path.join(self.root, "IMAGES", fname)
        img = Image.open(img_path)
        if self.transforms_pytorch is not None:
            img = self.transforms_pytorch(img)

        sample = {}
        sample["idx"] = index
        sample["lab"] = label
        sample["img"] = img
        sample["fname"] = fname
        return sample

    def __len__(self):
        return len(self.index_file)


class Messidor:

    num_classes = 5
    input_size = (224, 224)
    patch_size = (16, 16)
    in_channels = 3
    multi_label = False

    def __init__(
        self,
        path_to_messidor: str = "/project/dane2/wficai/BenchMD/messidor",
        batch_size: int = 64,
        num_workers: int = 4,
        transforms_pytorch: str | transforms.Compose = "default",
        gpu=None,
        train_split_frac: float = 0.8,
    ):
        self.path_to_messidor = path_to_messidor
        self.transforms = transforms_pytorch
        self.gpu = gpu
        self.train_split_frac = train_split_frac
        self.root = path_to_messidor
        self.num_workers = num_workers
        self.batch_size = batch_size

    def edit_ext(self, row):
        if row["image_id"][-3:] == "jpg":
            row["image_id"] = row["image_id"][:-3] + "JPG"
        return row

    def build_index(self):
        print("Building index...")
        image_info = os.path.join(self.root, "messidor_data.csv")

        # Load image_id and adjudicated_dr_drade columns
        df = pd.read_csv(image_info, header=0, usecols=[0, 1])

        # Clean data
        df = df.dropna()  # Remove rows with missing diagnosis info
        df = df.apply(
            lambda row: self.edit_ext(row), axis=1
        )  # Standardize image extensions

        # Split data into training and validation sets
        unique_counts = df["adjudicated_dr_grade"].value_counts()
        train_df = pd.DataFrame(columns=df.columns)
        valid_df = pd.DataFrame(columns=df.columns)

        for label, count in unique_counts.items():
            # Determine sample size for training
            num_samples = int(self.train_split_frac * count)

            # Split data by label
            graded_rows = df.loc[df["adjudicated_dr_grade"] == label]

            # Sample training rows
            train_rows = graded_rows.sample(num_samples, random_state=42)

            # Update training and validation dataframes
            train_df = pd.concat([train_df, train_rows])
            valid_df = pd.concat([valid_df, graded_rows.drop(train_rows.index)])

        # Save training df
        train_df = train_df.reset_index()
        train_df_path = os.path.join(self.root, "train_index.csv")
        print(f"Saving train index to {train_df_path}")
        train_df.to_csv(train_df_path)

        # Save validation df
        valid_df = valid_df.reset_index()
        valid_df_path = os.path.join(self.root, "valid_index.csv")
        print(f"Saving valid index to {valid_df_path}")
        valid_df.to_csv(valid_df_path)

        print("Done \n\n")

    def build_index_cv(self, k: int = 4):
        """
        Builds k-fold cross-validation indices with balanced class distribution using sklearn's StratifiedKFold.
        Saves the splits in cv_splits folder as {fold}_{split}_index.csv
        
        Args:
            k (int): Number of folds for cross-validation. Defaults to 4.
        """
        print(f"Building {k}-fold cross-validation indices...")
        
        # Create cv_splits directory if it doesn't exist
        cv_splits_dir = os.path.join(self.root, "cv_splits")
        os.makedirs(cv_splits_dir, exist_ok=True)
        
        # Load and clean data
        image_info = os.path.join(self.root, "messidor_data.csv")
        df = pd.read_csv(image_info, header=0, usecols=[0, 1])
        df = df.dropna()
        df = df.apply(lambda row: self.edit_ext(row), axis=1)
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        # Get features (X) and labels (y)
        X = df.index.values
        y = df["adjudicated_dr_grade"].values
        
        # Generate and save splits
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            # Create train and validation dataframes
            train_df = df.iloc[train_idx].reset_index()
            valid_df = df.iloc[valid_idx].reset_index()
            
            # Save files
            train_path = os.path.join(cv_splits_dir, f"fold{fold}_train_index.csv")
            valid_path = os.path.join(cv_splits_dir, f"fold{fold}_valid_index.csv")
            
            print(f"Saving fold {fold} train index to {train_path}")
            print(f"Saving fold {fold} valid index to {valid_path}")
            
            # Save class distribution information
            train_dist = train_df["adjudicated_dr_grade"].value_counts().sort_index()
            valid_dist = valid_df["adjudicated_dr_grade"].value_counts().sort_index()
            print(f"\nFold {fold} class distribution:")
            print(f"Train: {dict(train_dist)}")
            print(f"Valid: {dict(valid_dist)}\n")
            
            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)
        
        print("Cross-validation splits created successfully\n")

    def get_dataset(self, split: str = "train", fold: int | None = None) -> MessidorBase:
        """
        Get dataset for the specified split.
        
        Args:
            split (str): Either "train" or "valid". Defaults to "train".
            fold (int, optional): If provided, uses cross-validation splits from cv_splits folder.
                                 If None, uses standard train/valid splits from root directory.
        
        Returns:
            MessidorBase: The dataset instance.
        """
        if fold is not None:
            # Use CV splits
            cv_splits_dir = os.path.join(self.root, "cv_splits")
            index_file_path = os.path.join(cv_splits_dir, f"fold{fold}_{split}_index.csv")
            
            if not os.path.exists(index_file_path):
                raise FileNotFoundError(
                    f"CV split file not found: {index_file_path}. "
                    f"Make sure to run build_index_cv() first."
                )
            
            d_messidor = MessidorBase(
                self.root, 
                split=split, 
                transforms_pytorch=self.transforms,
                index_file_path=index_file_path
            )
        else:
            # Use standard splits
            if split == "train":
                d_messidor = MessidorBase(
                    self.root, split="train", transforms_pytorch=self.transforms
                )
            elif split == "valid":
                d_messidor = MessidorBase(
                    self.root, split="valid", transforms_pytorch=self.transforms
                )
            else:
                raise ValueError(f"Invalid split: {split}")
        
        print(f"length of dataset {split}" + (f" (fold {fold})" if fold is not None else "") + f": {len(d_messidor)}")
        return d_messidor

    def check_dataset(self, idx: int = 1, split: str = "train", fold: int | None = None) -> None:
        # Get idx
        instance = self.get_dataset(split=split, fold=fold)[idx]
        # Print labels
        print(f"labels: {instance['lab']}")
        try:
            # Show image
            plt.imshow(instance["img"].squeeze())
        except:
            # Transpose dimensions if necessary
            plt.imshow(instance["img"].transpose(2, 0))

    def get_dataloader(self, split: str = "train", fold: int | None = None) -> torch.utils.data.DataLoader:
        """
        Get dataloader for the specified split.
        
        Args:
            split (str): Either "train" or "valid". Defaults to "train".
            fold (int, optional): If provided, uses cross-validation splits from cv_splits folder.
                                 If None, uses standard train/valid splits from root directory.
        
        Returns:
            torch.utils.data.DataLoader: The dataloader instance.
        """
        d_messidor = self.get_dataset(split, fold=fold)
        dataloader = torch.utils.data.DataLoader(
            d_messidor,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def check_dataloader(self, split: str = "train", fold: int | None = None) -> None:
        print(next(iter(self.get_dataloader(split, fold=fold))))

    def calculate_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        loss_func = torch.nn.CrossEntropyLoss().cuda(self.gpu)
        loss = loss_func(predictions, targets.cuda(self.gpu, non_blocking=True).long())
        return loss

    def calculate_auc(self, outputs, targets) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate AUROC
        outputs = torch.stack(outputs)
        targets = torch.stack(targets).long()

        au_roc = AUROC(task="multiclass", num_classes=self.num_classes, average=None)
        au_roc_macro = AUROC(task="multiclass", num_classes=self.num_classes, average="macro")
        au_roc_weighted = AUROC(task="multiclass",  average="weighted", num_classes=self.num_classes)

        auc_calc_all = au_roc(outputs, targets)
        auc_calc_macro = au_roc_macro(outputs, targets)
        auc_calc_weighted = au_roc_weighted(outputs, targets)

        return auc_calc_all, auc_calc_macro, auc_calc_weighted

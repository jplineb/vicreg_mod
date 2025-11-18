import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchmetrics import AUROC
from matplotlib import pyplot as plt
from PIL import Image
from torchxrayvision.datasets import xrv
from sklearn.model_selection import StratifiedKFold


class MRNetBase(VisionDataset):
    """
    A dataset class for the MRNet dataset, knee MRI dataset for ACL and meniscus tear detection.
    The dataset consists of 1,370 knee MRI exams with 1,104 (80.6%) abnormal exams.
    """
    
    num_classes = 2  # Binary classification for each task
    input_size = (224, 224)
    patch_size = (16, 16)
    in_channels = 3  # RGB images (converted from grayscale)
    training_label = "abnormal"
    training_view = "axial"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms_pytorch: str | None | transforms.Compose = "txrv",
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
    def transforms_pytorch(self):
        if self._transforms_pytorch == "manual":
            return transforms.Compose([
                transforms.Lambda(lambda img: Image.fromarray(img, mode='L')),  # Convert numpy array to PIL Image
                transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to RGB
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalization
            ])
        if self._transforms_pytorch == "txrv":
            return transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(self.input_size[0]),
                xrv.datasets.ToPILImage(),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ])
        elif self._transforms_pytorch is None:
            raise ValueError("Must pass transforms")

    def __getitem__(self, index):
        df_row = self.index_file.iloc[index]
        pnum = df_row["pnum"]
        
        # Load labels
        lab = df_row[self.training_label]

        # Load numpy arrays for each view
        npy_path = df_row[self.training_view]
        if os.path.exists(npy_path):
            # Load numpy array
            view_array = np.load(npy_path)
            
            # Apply transforms (PIL conversion is handled in transforms)
            if self._transforms_pytorch is "manual":
                view_array = view_array[len(view_array)//2]
                # view_array = view_array.squeeze(0)
                img = self.transforms_pytorch(view_array)
            elif self._transforms_pytorch is "txrv":
                img = self.transforms_pytorch(view_array)
        else:          
            raise ValueError(f"View '{self.training_view}' not found in index file for patient {pnum}")

        sample = {}
        sample["idx"] = index
        sample["pnum"] = pnum
        sample["lab"] = lab
        sample["view"] = self.training_view
        sample["img"] = img
        
        return sample

    def __len__(self):
        return len(self.index_file)


class MRNet:
    """MRNet dataset consists of 1,370 knee MRI exams performed at Stanford University Medical Center.
    The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports.
    The most common indications for the knee MRI examinations in this study included acute and chronic pain, follow-up or preoperative evaluation, injury/trauma.
    Examinations were performed with GE scanners (GE Discovery, GE Healthcare, Waukesha, WI) with standard knee MRI coil and a routine non-contrast knee MRI protocol that included the following sequences: coronal T1 weighted, coronal T2 with fat saturation, sagittal proton density (PD) weighted, sagittal T2 with fat saturation, and axial PD weighted with fat saturation.
    A total of 775 (56.6%) examinations used a 3.0-T magnetic field; the remaining used a 1.5-T magnetic field.
    """
    
    num_classes = 2  # Binary classification for each task
    multi_label = False  # Multiple binary classification tasks
    
    classes_a = ["ACL", "Meniscus"]
    classes_b = ["Normal", "Abnormal"]
    views = ["axial", "sagittal", "coronal"]

    def __init__(
        self,
        path_to_mrnet: str = "/project/dane2/wficai/BenchMD/MRNet/MRNet-v1.0",
        batch_size: int = 64,
        num_workers: int = 4,
        transforms_pytorch: str | transforms.Compose = "manual",
        gpu=None,
    ):
        self.path_to_mrnet = path_to_mrnet
        self.transforms = transforms_pytorch
        self.gpu = gpu
        self.root = path_to_mrnet
        self.num_workers = num_workers
        self.batch_size = batch_size

    def build_index(self):
        print("Building index...")

        print(f"Building training index")
        abnormal_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-abnormal.csv"), header=None, names=["pnum", "abnormal"])
        acl_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-acl.csv"), header=None, names=["pnum", "acl"])
        meniscus_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-meniscus.csv"), header=None, names=["pnum", "meniscus"])
        # Merge on pnum to deduplicate and combine all labels
        training_df = abnormal_df.merge(acl_df, on="pnum", how="outer").merge(meniscus_df, on="pnum", how="outer")

        # Ensure pnum is a string and 4 digits, padding as needed
        training_df['pnum'] = training_df['pnum'].apply(lambda x: str(x).zfill(4))
        training_df['axial'] = training_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "train", "axial", f"{x}.npy"))
        training_df['sagittal'] = training_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "train", "sagittal", f"{x}.npy"))
        training_df['coronal'] = training_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "train", "coronal", f"{x}.npy"))
        # Remove any duplicate pnum rows (shouldn't be any after merge, but just in case)
        training_df = training_df.drop_duplicates(subset=["pnum"])
        final_training_path = os.path.join(self.path_to_mrnet, "train_index.csv")
        training_df.to_csv(final_training_path, index=False)
        print(f"Saving train index to {final_training_path}")

        print(f"Building validation index")
        abnormal_df = pd.read_csv(os.path.join(self.path_to_mrnet, "valid-abnormal.csv"), header=None, names=["pnum", "abnormal"])
        acl_df = pd.read_csv(os.path.join(self.path_to_mrnet, "valid-acl.csv"), header=None, names=["pnum", "acl"])
        meniscus_df = pd.read_csv(os.path.join(self.path_to_mrnet, "valid-meniscus.csv"), header=None, names=["pnum", "meniscus"])
        # Merge on pnum to deduplicate and combine all labels
        validation_df = abnormal_df.merge(acl_df, on="pnum", how="outer").merge(meniscus_df, on="pnum", how="outer")

        # Ensure pnum is a string and 4 digits, padding as needed
        validation_df['pnum'] = validation_df['pnum'].apply(lambda x: str(x).zfill(4))
        validation_df['axial'] = validation_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "valid", "axial", f"{x}.npy"))
        validation_df['sagittal'] = validation_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "valid", "sagittal", f"{x}.npy"))
        validation_df['coronal'] = validation_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "valid", "coronal", f"{x}.npy"))
        # Remove any duplicate pnum rows (shouldn't be any after merge, but just in case)
        validation_df = validation_df.drop_duplicates(subset=["pnum"])
        final_validation_path = os.path.join(self.path_to_mrnet, "valid_index.csv")
        validation_df.to_csv(final_validation_path, index=False)
        print(f"Saving valid index to {final_validation_path}")
        print("Done")

    def build_index_cv(self, k: int = 4):
        """
        Builds k-fold cross-validation indices with balanced class distribution using sklearn's StratifiedKFold.
        Creates combined splits (like build_index) with all tasks, stratified on 'abnormal' label.
        Saves the splits in cv_splits folder as fold{fold}_{split}_index.csv
        
        Args:
            k (int): Number of folds for cross-validation. Defaults to 4.
        """
        print(f"Building {k}-fold cross-validation indices...")
        
        # Create cv_splits directory if it doesn't exist
        cv_splits_dir = os.path.join(self.root, "cv_splits")
        os.makedirs(cv_splits_dir, exist_ok=True)
        
        # Load all task data (same as build_index)
        abnormal_train_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-abnormal.csv"), header=None, names=["pnum", "abnormal"])
        acl_train_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-acl.csv"), header=None, names=["pnum", "acl"])
        meniscus_train_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-meniscus.csv"), header=None, names=["pnum", "meniscus"])
        
        abnormal_valid_df = pd.read_csv(os.path.join(self.path_to_mrnet, "valid-abnormal.csv"), header=None, names=["pnum", "abnormal"])
        acl_valid_df = pd.read_csv(os.path.join(self.path_to_mrnet, "valid-acl.csv"), header=None, names=["pnum", "acl"])
        meniscus_valid_df = pd.read_csv(os.path.join(self.path_to_mrnet, "valid-meniscus.csv"), header=None, names=["pnum", "meniscus"])
        
        # Combine train and valid for each task
        abnormal_combined = pd.concat([abnormal_train_df, abnormal_valid_df], ignore_index=True)
        acl_combined = pd.concat([acl_train_df, acl_valid_df], ignore_index=True)
        meniscus_combined = pd.concat([meniscus_train_df, meniscus_valid_df], ignore_index=True)
        
        # Merge all tasks on pnum (same as build_index)
        combined_df = abnormal_combined.merge(acl_combined, on="pnum", how="outer").merge(meniscus_combined, on="pnum", how="outer")
        combined_df = combined_df.dropna(subset=["abnormal"])  # Keep only rows with abnormal label
        
        # Ensure pnum is a string and 4 digits, padding as needed
        combined_df['pnum'] = combined_df['pnum'].apply(lambda x: str(x).zfill(4))
        
        # Determine which directory (train or valid) each pnum belongs to
        train_pnums = set(abnormal_train_df['pnum'].apply(lambda x: str(x).zfill(4)))
        
        # Initialize StratifiedKFold (stratify on abnormal label)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        # Get features (X) and labels (y) for stratification
        X = combined_df.index.values
        y = combined_df["abnormal"].values # Build index with balanced class distribution abnormal view
        
        # Generate and save splits
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            # Create train and validation dataframes
            train_df = combined_df.iloc[train_idx].reset_index(drop=True)
            valid_df = combined_df.iloc[valid_idx].reset_index(drop=True)
            
            # Add view paths (determine directory based on original train/valid split)
            for view in self.views:
                train_df[view] = train_df['pnum'].apply(
                    lambda x: os.path.join(self.root, "train" if x in train_pnums else "valid", view, f"{x}.npy")
                )
                valid_df[view] = valid_df['pnum'].apply(
                    lambda x: os.path.join(self.root, "train" if x in train_pnums else "valid", view, f"{x}.npy")
                )
            
            # Save files
            train_path = os.path.join(cv_splits_dir, f"fold{fold}_train_index.csv")
            valid_path = os.path.join(cv_splits_dir, f"fold{fold}_valid_index.csv")
            
            print(f"Saving fold {fold} train index to {train_path}")
            print(f"Saving fold {fold} valid index to {valid_path}")
            
            # Save class distribution information
            train_dist = train_df["abnormal"].value_counts().sort_index()
            valid_dist = valid_df["abnormal"].value_counts().sort_index()
            print(f"\nFold {fold} class distribution (abnormal):")
            print(f"Train: {dict(train_dist)}")
            print(f"Valid: {dict(valid_dist)}\n")
            
            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)
        
        print("Cross-validation splits created successfully\n")

    def get_dataset(self, split: str = "train", fold: int | None = None) -> MRNetBase:
        """
        Get dataset for the specified split.
        
        Args:
            split (str): Either "train" or "valid". Defaults to "train".
            fold (int, optional): If provided, uses cross-validation splits from cv_splits folder.
                                 If None, uses standard train/valid splits from root directory.
        
        Returns:
            MRNetBase: The dataset instance.
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
            
            d_mrnet = MRNetBase(
                self.root,
                split=split,
                transforms_pytorch=self.transforms,
                index_file_path=index_file_path
            )
        else:
            # Use standard splits
            if split == "train":
                d_mrnet = MRNetBase(
                    self.root, split="train", transforms_pytorch=self.transforms
                )
            elif split == "valid":
                d_mrnet = MRNetBase(
                    self.root, split="valid", transforms_pytorch=self.transforms
                )
            else:
                raise ValueError(f"Invalid split: {split}")
        
        print(f"length of dataset {split}" + (f" (fold {fold})" if fold is not None else "") + f": {len(d_mrnet)}")
        return d_mrnet

    def check_dataset(self, idx: int = 1, split: str = "train", fold: int | None = None) -> None:
        """
        Visualizes the three knee MRI views for the sample at index `idx` in the provided split/fold.
        Processes the dict returned from __getitem__ properly (expects keys: 'img', 'lab', 'idx', 'pnum').
        """
        instance = self.get_dataset(split=split, fold=fold)[idx]
        
        # "instance" is expected to be a dict from __getitem__, containing:
        # 'img' -- the knee image corresponding to the selected view (probably just one view, e.g. 'axial')
        # 'lab' -- label
        # 'idx' -- index
        # 'pnum' -- patient number

        print(f"Entry idx: {instance.get('idx', idx)}, Patient #: {instance.get('pnum', None)}, Label: {instance.get('lab', None)}")

        # Try to visualize all three views: need to load them if not already present.
        # The dataset instance is tied to `self.training_view`; load others dynamically.
        dataset = self.get_dataset(split=split, fold=fold)
        # Use same row from index_file to get other views
        if hasattr(dataset, "index_file"):
            df_row = dataset.index_file.iloc[idx]
            base_sample = dict(instance)  # copy to avoid mutating
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            views = ["axial", "sagittal", "coronal"]
            for i, view in enumerate(views):
                npy_path = df_row.get(view, None)
                if (npy_path is not None) and (isinstance(npy_path, str)) and os.path.exists(npy_path):
                    arr = np.load(npy_path)
                    # Middle slice
                    if arr.ndim == 3:
                        arr = arr[len(arr) // 2]
                    # Now arr should be 2D, convert and apply appropriate transforms for display
                    img_disp = arr
                    if arr.ndim == 2:
                        # Use the dataset's transforms if possible
                        if hasattr(dataset, '_transforms_pytorch') and dataset._transforms_pytorch is not None:
                            transforms_func = dataset.transforms_pytorch
                        else:
                            transforms_func = None
                        arr_input = arr
                        # Some transforms expect a 3D stack for "manual", so mimic __getitem__ logic
                        if hasattr(dataset, '_transforms_pytorch') and dataset._transforms_pytorch == "manual":
                            # The transform expects a 2D slice as input
                            img_disp = transforms_func(arr_input)
                        elif hasattr(dataset, '_transforms_pytorch') and dataset._transforms_pytorch == "txrv":
                            # txrv expects a 3D (stack) input
                            arr_stack = arr_input[None, ...]  # shape (1, H, W)
                            img_disp = transforms_func(arr_stack)
                        else:
                            img_disp = arr_input
                        # Move to numpy for plt.imshow
                        if torch.is_tensor(img_disp):
                            if img_disp.ndim == 3 and img_disp.shape[0] in [1, 3]:
                                # C,H,W to H,W,C
                                img_disp = img_disp.permute(1, 2, 0).cpu().numpy()
                            else:
                                img_disp = img_disp.cpu().numpy()
                        if img_disp.dtype != np.float32:
                            img_disp = img_disp.astype(np.float32)
                        # img_disp = (img_disp - img_disp.min()) / (img_disp.ptp() + 1e-5)
                    axes[i].imshow(img_disp, cmap='gray' if img_disp.shape[-1] == 1 else None)
                    axes[i].set_title(f"{view.capitalize()} view")
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f"No {view} data", ha='center', va='center', fontsize=14)
                    axes[i].set_title(f"{view.capitalize()} view")
                    axes[i].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            # Fallback option: only show the loaded image tensor
            img_tensor = instance.get("img", None)
            if img_tensor is not None:
                if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
                    img_disp = img_tensor.permute(1, 2, 0).cpu().numpy()
                    plt.imshow(img_disp)
                elif img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
                    img_disp = img_tensor.squeeze(0).cpu().numpy()
                    plt.imshow(img_disp, cmap='gray')
                else:
                    plt.imshow(img_tensor.cpu().numpy(), cmap='gray')
                plt.title("Loaded image view")
                plt.axis('off')
                plt.show()
            else:
                print("No image tensor found to display.")

        return instance

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
        d_mrnet = self.get_dataset(split, fold=fold)
        dataloader = torch.utils.data.DataLoader(
            d_mrnet,
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
        """
        Calculate loss for binary classification.
        """
        # For binary classification with BCEWithLogitsLoss:
        # - predictions should be (N,) or (N, 1) - logits for positive class
        # - targets should be (N,) - binary labels (0 or 1)
        
        # If predictions are (N, 2), take only the positive class logits
        if predictions.shape[1] == 2:
            predictions = predictions[:, 1]  # Take positive class logits
        
        # Ensure targets are float and on correct device
        targets = targets.float().cuda(self.gpu, non_blocking=True)
        
        loss_func = torch.nn.BCEWithLogitsLoss().cuda(self.gpu)
        loss = loss_func(predictions, targets)
        return loss

    def calculate_auc(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate AUROC for binary classification
        """
        # Calculate AUROC
        outputs = torch.stack(outputs)
        targets = torch.stack(targets).long()
        
        # For binary classification with AUROC:
        # - outputs should be (N,) - probabilities/logits for positive class
        # - targets should be (N,) - binary labels (0 or 1)
        
        # If outputs are (N, 2), take only the positive class probabilities/logits
        if outputs.shape[1] == 2:
            outputs = outputs[:, 1]  # Take positive class probabilities/logits

        au_roc = AUROC(task="binary", average=None)
        au_roc_macro = AUROC(task="binary", average="macro")
        au_roc_weighted = AUROC(task="binary", average="weighted")

        auc_calc_all = au_roc(outputs, targets)
        auc_calc_macro = au_roc_macro(outputs, targets)
        auc_calc_weighted = au_roc_weighted(outputs, targets)

        return auc_calc_all, auc_calc_macro, auc_calc_weighted

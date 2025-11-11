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
    ):
        self._transforms_pytorch = transforms_pytorch
        assert self._transforms_pytorch is not None
        super().__init__(root, transform=self.transforms_pytorch)

        self.split = split
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
        Saves the splits in cv_splits folder as {fold}_{split}_index.csv
        
        Args:
            k (int): Number of folds for cross-validation. Defaults to 4.
        """
        from sklearn.model_selection import StratifiedKFold
        
        print(f"Building {k}-fold cross-validation indices...")
        
        # Create cv_splits directory if it doesn't exist
        cv_splits_dir = os.path.join(self.root, "cv_splits")
        os.makedirs(cv_splits_dir, exist_ok=True)
        
        # Load and clean data for each task
        tasks = ['abnormal', 'acl', 'meniscus']
        
        for task in tasks:
            print(f"Building CV splits for {task} task...")
            
            # Load task-specific data
            train_file = os.path.join(self.root, f"train-{task}.csv")
            valid_file = os.path.join(self.root, f"valid-{task}.csv")
            
            if os.path.exists(train_file) and os.path.exists(valid_file):
                train_df = pd.read_csv(train_file, header=None, names=["pnum", task])
                valid_df = pd.read_csv(valid_file, header=None, names=["pnum", task])
                
                # Combine train and valid for CV
                combined_df = pd.concat([train_df, valid_df], ignore_index=True)
                combined_df = combined_df.dropna()
                
                # Initialize StratifiedKFold
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                
                # Get features (X) and labels (y)
                X = combined_df.index.values
                y = combined_df[task].values
                
                # Generate and save splits
                for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
                    # Create train and validation dataframes
                    train_fold_df = combined_df.iloc[train_idx].reset_index()
                    valid_fold_df = combined_df.iloc[valid_idx].reset_index()
                    
                    # Add view paths
                    for view in self.views:
                        train_fold_df[view] = train_fold_df['pnum'].apply(
                            lambda x: os.path.join(self.root, "train" if x in train_df['pnum'].values else "valid", view, f"{x}.npy")
                        )
                        valid_fold_df[view] = valid_fold_df['pnum'].apply(
                            lambda x: os.path.join(self.root, "train" if x in train_df['pnum'].values else "valid", view, f"{x}.npy")
                        )
                    
                    # Save files
                    train_path = os.path.join(cv_splits_dir, f"fold{fold}_{task}_train_index.csv")
                    valid_path = os.path.join(cv_splits_dir, f"fold{fold}_{task}_valid_index.csv")
                    
                    print(f"Saving fold {fold} {task} train index to {train_path}")
                    print(f"Saving fold {fold} {task} valid index to {valid_path}")
                    
                    # Save class distribution information
                    train_dist = train_fold_df[task].value_counts().sort_index()
                    valid_dist = valid_fold_df[task].value_counts().sort_index()
                    print(f"\nFold {fold} {task} class distribution:")
                    print(f"Train: {dict(train_dist)}")
                    print(f"Valid: {dict(valid_dist)}\n")
                    
                    train_fold_df.to_csv(train_path, index=False)
                    valid_fold_df.to_csv(valid_path, index=False)
        
        print("Cross-validation splits created successfully\n")

    def get_dataset(self, split: str = "train") -> MRNetBase:
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
        print(f"length of dataset {split}: {len(d_mrnet)}")
        return d_mrnet

    def check_dataset(self, idx: int = 1, split: str = "train") -> None:
        # Get idx
        instance = self.get_dataset(split=split)[idx]
        
        # Show all three views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        views = ['axial', 'sagittal', 'coronal']
        
        for i, view in enumerate(views):
            if view in instance:
                # Denormalize the tensor for visualization
                img_tensor = instance[view]
                if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
                    # RGB tensor: convert from (C, H, W) to (H, W, C)
                    img_tensor = img_tensor.permute(1, 2, 0)
                elif img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
                    # Grayscale tensor: squeeze to (H, W)
                    img_tensor = img_tensor.squeeze(0)
                
                # For RGB, show as is; for grayscale, use gray colormap
                if img_tensor.shape[-1] == 3:
                    axes[i].imshow(img_tensor.numpy())
                else:
                    axes[i].imshow(img_tensor.numpy(), cmap='gray')
                axes[i].set_title(f"{view.capitalize()} view")
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"No {view} data", ha='center', va='center')
                axes[i].set_title(f"{view.capitalize()} view")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return instance

    def get_dataloader(self, split: str = "train") -> torch.utils.data.DataLoader:
        d_mrnet = self.get_dataset(split)
        dataloader = torch.utils.data.DataLoader(
            d_mrnet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def check_dataloader(self, split: str = "train") -> None:
        print(next(iter(self.get_dataloader(split))))

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

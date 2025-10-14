import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class MRNetBase(VisionDataset):

    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1
    NUM_CLASSES = 2
    MULTI_LABEL = False

    def __init__(self, path_to_mrnet: str = "/project/dane2/wficai/BenchMD/MRNet/MRNet-v1.0"):
        self.path_to_mrnet = path_to_mrnet

        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
            ]
        )

    def build_index(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    @staticmethod
    def num_classes():
        pass

    @staticmethod
    def spec():
        pass

class MRNet:
    """MRNet dataset consists of 1,370 knee MRI exams performed at Stanford University Medical Center.
    The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports.\
    The most common indications for the knee MRI examinations in this study included acute and chronic pain, follow-up or preoperative evaluation, injury/trauma.
    Examinations were performed with GE scanners (GE Discovery, GE Healthcare, Waukesha, WI) with standard knee MRI coil and a routine non-contrast knee MRI protocol that included the following sequences: coronal T1 weighted, coronal T2 with fat saturation, sagittal proton density (PD) weighted, sagittal T2 with fat saturation, and axial PD weighted with fat saturation.
    A total of 775 (56.6%) examinations used a 3.0-T magnetic field; the remaining used a 1.5-T magnetic field.
    """

    classes_a = ["ACL", "Meniscus"]
    classes_b = ["Normal", "Abnormal"]

    views = ["axial", "sagittal", "coronal"]

    def __init__(
        self,
        path_to_mrnet: str = "/project/dane2/wficai/BenchMD/MRNet/MRNet-v1.0",
        transforms_pytorch: str = "default",
        data_mode: str = "files",  # "files" or "numpy"
        numpy_data: dict = None,  # Dictionary containing numpy arrays when data_mode="numpy"
    ):
        self.path_to_mrnet = path_to_mrnet
        self.transforms = transforms_pytorch
        self.data_mode = data_mode
        self.numpy_data = numpy_data
        
        # Define transforms for numpy arrays
        self.numpy_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
        ])

    def build_index(self):
        print("Building indexii")
        print(f"Building training index")
        abnormal_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-abnormal.csv"), header=None, names=["pnum", "abnormal"])
        acl_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-acl.csv"), header=None, names=["pnum", "acl"])
        meniscus_df = pd.read_csv(os.path.join(self.path_to_mrnet, "train-meniscus.csv"), header=None, names=["pnum", "meniscus"])
        # Merge on pnum to deduplicate and combine all labels
        training_df = abnormal_df.merge(acl_df, on="pnum", how="outer").merge(meniscus_df, on="pnum", how="outer")
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
        validation_df['axial'] = validation_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "valid", "axial", f"{x}.npy"))
        validation_df['sagittal'] = validation_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "valid", "sagittal", f"{x}.npy"))
        validation_df['coronal'] = validation_df['pnum'].apply(lambda x: os.path.join(self.path_to_mrnet, "valid", "coronal", f"{x}.npy"))
        # Remove any duplicate pnum rows (shouldn't be any after merge, but just in case)
        validation_df = validation_df.drop_duplicates(subset=["pnum"])
        final_validation_path = os.path.join(self.path_to_mrnet, "valid_index.csv")
        validation_df.to_csv(final_validation_path, index=False)
        print(f"Saving valid index to {final_validation_path}")
        print("Done")

    def get_dataset(self, split: str = "train"):
        if self.data_mode == "numpy":
            return self._get_numpy_dataset(split)
        else:
            return self._get_file_dataset(split)
    
    def _get_numpy_dataset(self, split: str = "train"):
        """Return dataset when using numpy arrays as input"""
        if self.numpy_data is None:
            raise ValueError("numpy_data must be provided when data_mode='numpy'")
        
        # Return a simple dataset that uses the provided numpy arrays
        class NumpyDataset:
            def __init__(self, numpy_data, transforms, views):
                self.data = numpy_data
                self.transforms = transforms
                self.views = views
                
            def __getitem__(self, idx):
                # Get the data for this index
                sample = self.data[idx]
                
                # Process each view
                processed_views = {}
                for view in self.views:
                    if view in sample:
                        # Convert numpy array to tensor and apply transforms
                        if isinstance(sample[view], np.ndarray):
                            # Ensure the array is in the right format (H, W, C) for ToPILImage
                            if len(sample[view].shape) == 3 and sample[view].shape[0] == 1:
                                # If shape is (1, H, W), transpose to (H, W, 1)
                                view_data = sample[view].squeeze(0).transpose(1, 2, 0)
                            elif len(sample[view].shape) == 2:
                                # If shape is (H, W), add channel dimension
                                view_data = np.expand_dims(sample[view], axis=2)
                            else:
                                view_data = sample[view]
                            
                            # Convert to PIL Image and apply transforms
                            if view_data.dtype != np.uint8:
                                # Normalize to 0-255 range if needed
                                view_data = ((view_data - view_data.min()) / (view_data.max() - view_data.min()) * 255).astype(np.uint8)
                            
                            processed_views[view] = self.transforms(view_data)
                        else:
                            processed_views[view] = sample[view]
                
                # Return labels if available
                labels = {}
                if 'abnormal' in sample:
                    labels['abnormal'] = torch.tensor(sample['abnormal'], dtype=torch.float32)
                if 'acl' in sample:
                    labels['acl'] = torch.tensor(sample['acl'], dtype=torch.float32)
                if 'meniscus' in sample:
                    labels['meniscus'] = torch.tensor(sample['meniscus'], dtype=torch.float32)
                
                return processed_views, labels
            
            def __len__(self):
                return len(self.data)
        
        return NumpyDataset(self.numpy_data, self.numpy_transforms, self.views)
    
    def _get_file_dataset(self, split: str = "train"):
        """Return dataset when loading from files (original behavior)"""
        # This would implement the original file-based loading
        # For now, return None as the original implementation was incomplete
        return None
    
    @classmethod
    def from_numpy_arrays(cls, numpy_data, transforms_pytorch: str = "default"):
        """
        Create an MRNet dataset from numpy arrays.
        
        Args:
            numpy_data: List of dictionaries, where each dictionary contains:
                - 'axial': numpy array of shape (H, W) or (1, H, W)
                - 'sagittal': numpy array of shape (H, W) or (1, H, W)  
                - 'coronal': numpy array of shape (H, W) or (1, H, W)
                - 'abnormal': int (0 or 1)
                - 'acl': int (0 or 1) 
                - 'meniscus': int (0 or 1)
            transforms_pytorch: Transform configuration
            
        Returns:
            MRNet dataset configured for numpy array input
        """
        return cls(
            data_mode="numpy",
            numpy_data=numpy_data,
            transforms_pytorch=transforms_pytorch
        )

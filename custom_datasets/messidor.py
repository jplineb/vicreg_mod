from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import torchxrayvision as xrv
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
from custom_datasets.dataset_utils.input_spec import Input2dSpec
from torchvision.datasets import VisionDataset


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
    ):
        self._transforms_pytorch = transforms_pytorch
        super().__init__(root, transform=self.transforms_pytorch)
        
        self.split = split
        self.index_file = pd.read_csv(
            os.path.join(self.root, f"{self.split}_index.csv")
        )

    @property
    def transforms_pytorch(self):
        if self._transforms_pytorch == "default":
            return transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize(self.input_size[0] - 1, max_size=self.input_size[0]),  # resizes (H,W) to (149, 224)
                transforms.ToTensor(),
                transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
                transforms.Pad((0, 37, 0, 38))
            ])
        elif self._transforms_pytorch is None:
            raise ValueError("Must pass transforms")
        else:
            return self._transforms_pytorch

    def __getitem__(self, index):
        df_row = self.index_file.iloc[index]
        fname = df_row['image_id']
        label = df_row['adjudicated_dr_grade']
        
        # Load image
        img_path = os.path.join(self.root, 'IMAGES', fname)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
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
    
    def __init__(
        self,
        path_to_messidor: str,
        batch_size: int = 64,
        num_workers: int = 4,
        transforms_pytorch="default",
        gpu = None,
        train_split_frac:float = 0.8
    ):  
        self.path_to_messidor = path_to_messidor
        self.transforms = transforms_pytorch
        self.gpu = gpu
        self.train_split_frac = train_split_frac
        self.root = path_to_messidor
        self.num_workers = num_workers
        self.batch_size = batch_size

    def edit_ext(self, row):
        if row['image_id'][-3:] == "jpg":
            row['image_id'] = row['image_id'][:-3] + 'JPG'
        return row
        
    def build_index(self):
        print('Building index...')
        image_info = os.path.join(self.root, 'messidor_data.csv')

        # Load image_id and adjudicated_dr_drade columns
        df = pd.read_csv(image_info, header=0, usecols=[0, 1])

        # Clean data
        df = df.dropna()  # Remove rows with missing diagnosis info
        df = df.apply(lambda row: self.edit_ext(row), axis=1)  # Standardize image extensions

        # Split data into training and validation sets
        unique_counts = df['adjudicated_dr_grade'].value_counts()
        train_df = pd.DataFrame(columns=df.columns)
        valid_df = pd.DataFrame(columns=df.columns)

        for label, count in unique_counts.items():
            # Determine sample size for training
            num_samples = int(self.train_split_frac * count)
            
            # Split data by label
            graded_rows = df.loc[df['adjudicated_dr_grade'] == label]
            
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

        print('Done \n\n')
    
    def get_dataset(self, split: str = "train"):
        if split == "train":
            d_messidor = MessidorBase(self.root, split="train", transforms_pytorch=self.transforms)
        elif split == "valid":
            d_messidor = MessidorBase(self.root, split="valid", transforms_pytorch=self.transforms)
        else:
            raise ValueError(f"Invalid split: {split}")
        print(f"length of dataset {split}: {len(d_messidor)}")
        return d_messidor
    
    def check_dataset(self, idx: int = 1, split: str = "train"):
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
    
    def get_dataloader(self, split: str = "train"):
        d_messidor = self.get_dataset(split)
        dataloader = torch.utils.data.DataLoader(d_messidor, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return dataloader
    
    def check_dataloader(self, split: str = "train"):
        print(next(iter(self.get_dataloader(split))))
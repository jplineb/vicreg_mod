# /home/ubuntu/raw/vindr/physionet.org/files/vindr-cxr/1.0.0
import glob
import os
import shutil
from typing import Any, Union, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from torch import nn

from torchmetrics import AUROC

CHEXPERT_LABELS = {
    "No Finding": 0,
    "Enlarged Cardiomediastinum": 1,
    "Cardiomegaly": 2,
    "Lung Opacity": 3,
    "Lung Lesion": 4,
    "Edema": 5,
    "Consolidation": 6,
    "Pneumonia": 7,
    "Atelectasis": 8,
    "Pneumothorax": 9,
    "Pleural Effusion": 10,
    "Pleural Other": 11,
    "Fracture": 12,
    "Support Devices": 13,
}


def count_files(dir_path):
    count = 0
    for _, _, files in os.walk(dir_path):
        count += len(files)
    return count


class Input2dSpec(object):
    """Defines the specs for 2d inputs."""

    input_type = "2d"

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        in_channels: int,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels


def any_exist(files):
    return any(map(os.path.exists, files))


class VINDR_CXR_BASE(VisionDataset):
    """A dataset class for the VINDR-CXR dataset (https://physionet.org/content/vindr-cxr/1.0.0/).
    Note that you must register and manually download the data to use this dataset.
    """

    # Dataset information.
    label_columns = [
        "Aortic enlargement",  # Enlarged Cardiomediastinum
        "Cardiomegaly",
        "Lung Opacity",
        "Lung cyst",  # Lung Lesion
        "Pulmonary fibrosis",  # Edema
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural effusion",
        "Pleural thickening",  # Plueral Other
        "Rib fracture",  # Fracture
        "Tuberculosis",  # Support Devices
    ]

    NUM_CLASSES = 13  # 14 total: len(self.CHEXPERT_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, train: bool = True, build_index=False) -> None:
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.Grayscale(3),  # John Paul added this
                transforms.ToTensor(),
                transforms.Normalize([0.7635], [0.1404]),
            ]
        )

        self.root = base_root
        super().__init__(self.root)
        # self.index_location = self.find_data()
        self.split = "train" if train else "test"
        if build_index is True:
            self.build_index()
        else:
            index_file = pd.read_csv(
                os.path.join(self.root, f"annotations/image_labels_{self.split}.csv")
            )

            index_file["fname"] = np.array(
                index_file["image_id"].apply(
                    lambda x: os.path.join(self.root, f"{self.split}/jpegs/{x}.jpg")
                )
            )

            # index_file.fnames = glob.glob(
            #     os.path.join(self.root, self.split + "/" + "jpegs") + "/*.jpg"
            # )

            # Drop duplicates, assume first is ground truth
            # self.index_file = self.index_file.drop_duplicates(
            #     subset=["fname"], keep="first"
            # )
            
            # Drop duplicates, use majority vote
            self.index_file = index_file.groupby(by=["fname"]).mean().round().reset_index()

    def read_dicom(self, file_path: str, imsize: int):
        """Read pixel array from a DICOM file and apply recale and resize
        operations.
        The rescale operation is defined as the following:
            x = x * RescaleSlope + RescaleIntercept
        The rescale slope and intercept can be found in the DICOM files.
        Args:
            file_path (str): Path to a dicom file.
            resize_shape (int): Height and width for resizing.
        Returns:
            The processed pixel array from a DICOM file.
        """

        # read dicom
        dcm = pydicom.dcmread(file_path)
        pixel_array = dcm.pixel_array

        # rescale
        if "RescaleIntercept" in dcm:
            intercept = dcm.RescaleIntercept
            slope = dcm.RescaleSlope
            pixel_array = pixel_array * slope + intercept

        return pixel_array

    def dicom_to_jpg(self, fnames, imsize):
        if not os.path.isdir(os.path.join(self.root, self.split + "/" + "jpegs")):
            os.makedirs(os.path.join(self.root, self.split + "/" + "jpegs"))
        for i in tqdm(range(len(fnames))):
            dicom_path = fnames[i] + ".dicom"
            # jpg_path = os.path.join(self.root, self.split + "/" + "jpegs/" + fnames[i].split("/")[-1] + ".jpg")
            # if not os.path.exists(jpg_path):
            img_array = self.read_dicom(dicom_path, imsize)
            img = Image.fromarray(img_array).convert("L")
            img.save(
                os.path.join(
                    self.root,
                    self.split + "/" + "jpegs/" + fnames[i].split("/")[-1] + ".jpg",
                )
            )

    def build_index(self):
        print("Building index...")

        metadata = pd.read_csv(
            os.path.join(self.root, f"annotations/image_labels_{self.split}.csv")
        )
        index_file = metadata

        dicom_fnames = np.array(
            index_file["image_id"].apply(
                lambda x: os.path.join(self.root, f"{self.split}/{x}")
            )
        )
        if self.split == "train":
            n_files = 15000
        else:
            n_files = 3000
        if (
            not os.path.isdir(os.path.join(self.root, self.split + "/" + "jpegs"))
        ) or count_files(os.path.join(self.root, self.split + "/" + "jpegs")) < n_files:
            if os.path.isdir(os.path.join(self.root, self.split + "/" + "jpegs")):
                shutil.rmtree(os.path.join(self.root, self.split + "/" + "jpegs"))
            self.dicom_to_jpg(fnames=dicom_fnames, imsize=self.INPUT_SIZE[0])
        self.fnames = glob.glob(
            os.path.join(self.root, self.split + "/" + "jpegs") + "/*.jpg"
        )
        LABELS_COL = index_file.columns.get_loc("Aortic enlargement")
        end_col = LABELS_COL + len(CHEXPERT_LABELS)
        # missing values occur when no comment is made on a particular diagnosis. we treat this as a negative diagnosis
        self.labels = index_file.iloc[:, range(LABELS_COL, end_col)].values
        self.labels = np.maximum(self.labels, 0)  # convert -1 (unknown) to 0
        print("Done")

    def __len__(self) -> int:
        return len(self.index_file)

    def __getitem__(self, index: int) -> Any:
        df_row = self.index_file.iloc[index]
        fname = df_row["fname"]
        image = Image.open(os.path.join(self.root, fname)).convert("L")
        img = self.TRANSFORMS(image)

        sample = {}
        sample["idx"] = index

        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            # edge case 223,223, resize to match 224*224
            dim_gap = self.INPUT_SIZE[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)
        label = torch.tensor([df_row[label] for label in self.label_columns]).long()

        sample["lab"] = label
        sample["img"] = img.float()

        return sample

    @staticmethod
    def num_classes():
        return VINDR_CXR.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=VINDR_CXR.INPUT_SIZE,
                patch_size=VINDR_CXR.PATCH_SIZE,
                in_channels=VINDR_CXR.IN_CHANNELS,
            ),
        ]


class VINDR_CXR:
    def __init__(
        self,
        path_to_vindcxr: str = "/project/dane2/wficai/BenchMD/vindr-cxr/1.0.0/",
        batch_size: int = 64,
        num_workers: int = 4,
        gpu=None,
    ):
        self.data_loader_spec = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        self.gpu = gpu

        self.path = path_to_vindcxr

        self.pathologies = [
            "Aortic enlargement",  # Enlarged Cardiomediastinum
            "Cardiomegaly",
            "Lung Opacity",
            "Lung cyst",  # Lung Lesion
            "Pulmonary fibrosis",  # Edema
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural effusion",
            "Pleural thickening",  # Plueral Other
            "Rib fracture",  # Fracture
            "Tuberculosis",  # Support Devices
        ]

        self.pathologies_of_interest = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural effusion",
        ]

    def get_dataset(self, split="train"):
        if split == "train":
            d_cxr = VINDR_CXR_BASE(self.path, train=True)
        if split == "valid":
            d_cxr = VINDR_CXR_BASE(self.path, train=False)
        print(f"length of dataset {split}: {len(d_cxr)}")
        return d_cxr

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
                try:
                    print(f"Len of datasets {len(dataset)}")
                except:
                    print("whoops")
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
        loss = loss_func(predictions, targets.cuda(self.gpu, non_blocking=True).float())
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

        # # Get the AUROCs of interest
        auc_dict = {}
        auc_of_interest = []
        for pathology, auc in zip(self.pathologies, auc_calc_all.tolist()):
            # Store in master dict
            auc_dict[pathology] = auc
            # If AUC in pathologies of interest put them in list
            if pathology in self.pathologies_of_interest:
                auc_of_interest.append(auc)

        # Get average of AUROC of interest
        auc_of_avg_interest = 0

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

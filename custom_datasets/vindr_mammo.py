import shutil
from torchvision import transforms
from torch.utils.data import VisionDataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import tqdm
from custom_datasets.dataset_utils.dicom import get_pixel_array


class VindrMammographyBase(VisionDataset):
    """A dataset class for the VinDR Mammography dataset. (https://www.physionet.org/content/vindr-mammo/1.0.0/)
    This dataset consists of left/right breast images from one of two views.
    Each breast image is categorized on the BIRAD 1-5 scale, which communicates findings on presence/severity of lesions.
    Note:
        1) You must register and manually download the data to this directory to use this dataset. Rename the folder you download "vindr".
        Directions are available at the bottom of the above link after you make an account and click to complete the data use agreement.
    """

    # Dataset information.
    NUM_CLASSES = 5
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1
    RANDOM_SEED = 0

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms_pytorch: str | None | transforms.Compose = "default",
    ):
        self._transforms_pytorch = transforms_pytorch
        assert self._transforms_pytorch is not None
        super().__init__(root, transform=self.transforms_pytorch)

        self.split = split
        self.index_file = pd.read_csv(
            os.path.join(self.root, f"{self.split}_index.csv")
        )

    @property
    def transforms_pytorch(self) -> transforms.Compose:
        if self._transforms_pytorch == "default":
            return transforms.Compose(
                [
                    transforms.Resize(
                        self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.1180], [1]),
                ]
            )
        elif self._transforms_pytorch is None:
            raise ValueError("Must pass transforms")

    def __getitem__(self, index):
        img_path = os.path.join(
            self.root, "jpegs", self.fnames[index][0], self.fnames[index][2] + ".jpg"
        )
        # Convert BIRAD 1-5 classification to class label (0-4)
        label = int(self.labels[index][-1]) - 1
        img = Image.open(img_path)
        img = self.transforms(img)

        # Add image-dependent padding
        dim_gap = img.shape[1] - img.shape[2]
        pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
        img = transforms.Pad((pad1, 0, pad2, 0))(img)

        return index, img, label

    def __len__(self):
        return len(self.index_file)

    @staticmethod
    def num_classes():
        return VindrMammographyBase.NUM_CLASSES


class VindrMammography:
    """A dataset class for the VinDR Mammography dataset. (https://www.physionet.org/content/vindr-mammo/1.0.0/)
    This dataset consists of left/right breast images from one of two views.
    Each breast image is categorized on the BIRAD 1-5 scale, which communicates findings on presence/severity of lesions.
    """

    def __init__(
        self,
        path_to_vindr_mammo: str = "/project/dane2/wficai/BenchMD/vindr-mammo",
        batch_size: int = 64,
        num_workers: int = 4,
        transforms_pytorch: str | transforms.Compose = "default",
        gpu=None,
        train_split_frac: float = 0.8,
    ):
        self.root = path_to_vindr_mammo
        self.transforms = transforms_pytorch
        self.gpu = gpu
        self.train_split_frac = train_split_frac
        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_dataset(self, split="train"):
        pass

    # save all dicom files as jpgs ahead of training for faster processing
    # def dicom_to_jpg(self, df):
    #     fnames = df.iloc[:, [0, 1, 2]].to_numpy(dtype=np.str)
    #     for i in tqdm.tqdm(range(len(fnames))):
    #         # ignore race condition errors
    #         try:
    #             if not os.path.isdir(os.path.join(self.root, "jpegs", fnames[i][0])):
    #                 os.makedirs(os.path.join(self.root, "jpegs", fnames[i][0]))
    #         except OSError as e:
    #             if e.errno != 17:
    #                 print("Error:", e)
    #         dicom_path = os.path.join(
    #             self.root, "images", fnames[i][0], fnames[i][2] + ".dicom"
    #         )
    #         img_array = get_pixel_array(dicom_path)
    #         img = Image.fromarray(img_array)
    #         img.save(
    #             os.path.join(self.root, "jpegs", fnames[i][0], fnames[i][2] + ".jpg")
    #         )

    # def build_index(self):
    #     print("Building index...")
    #     index_file = os.path.join(self.root, "breast-level_annotations.csv")

    #     # get columns for study_id, series_id, image_id, breast_birads, and split
    #     df = pd.read_csv(index_file, header=0, usecols=[0, 1, 2, 7, 9])

    #     print("Converting DICOM to JPG")
    #     # Convert DICOM files to JPGs
    #     if os.path.isdir(os.path.join(self.root, "jpegs")):
    #         if count_files(os.path.join(self.root, "jpegs")) != 20000:
    #             shutil.rmtree(os.path.join(self.root, "jpegs"))
    #             self.dicom_to_jpg(df)
    #     else:
    #         self.dicom_to_jpg(df)

    #     df = df.loc[df["split"] == self.split]  # use correct split

    #     # select subset of training data if finetuning
    #     if self.split == "training":
    #         # get counts for every label
    #         unique_counts = df.iloc[:, 3].value_counts()
    #         train_df = pd.DataFrame(columns=df.columns)

    #         for label, count in unique_counts.items():
    #             # get 'finetune_size' labels for each class
    #             # if insufficient examples, use all examples from that class
    #             num_sample = min(self.finetune_size, count)
    #             train_rows = df.loc[df.iloc[:, 3] == label].sample(
    #                 num_sample, random_state=VINDR.RANDOM_SEED
    #             )
    #             train_df = train_df.append(train_rows)

    #         df = train_df

    #     self.fnames = df.iloc[:, [0, 1, 2]].to_numpy(dtype=np.str)
    #     self.labels = df.iloc[:, 3].to_numpy(dtype=np.str)

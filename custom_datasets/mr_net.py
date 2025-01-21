from torchvision.datasets.vision import VisionDataset


class MRNetBase(VisionDataset):

    def __init__(self, path_to_mrnet: str = "/project/dane2/wficai/BenchMD/MRNet/"):
        self.path_to_mrnet = path_to_mrnet

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

    def __init__(
        self,
        path_to_mrnet: str = "/project/dane2/wficai/BenchMD/MRNet/",
        transforms_pytorch,
    ):
        pass

    def get_dataset(self, split="train"):
        pass

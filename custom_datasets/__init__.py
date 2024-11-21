from .chexpert import *
from .RadImageNet import *
from .mimic_cxr import *
from .vind_cxr import *
from .med_dataset import *


DATASETS = {
    "chexpert": Chexpert,
    "vinddrcxr": VINDR_CXR,
    "mimic_cxr": MIMIC_CXR,
    "radimagenet": RadImageNet
}

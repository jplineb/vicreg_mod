from .dataset_utils import *
from .chexpert import *
from .RadImageNet import *
from .mimic_cxr import *
from .vind_cxr import *
from .med_dataset import *
from .messidor import *
from .bcn_20000 import *

DATASETS = {
    "chexpert": Chexpert,
    "vinddrcxr": VINDR_CXR,
    "mimic_cxr": MIMIC_CXR,
    "radimagenet": RadImageNet,
    "messidor": Messidor,
    "bcn_20000": BCN20000,
}

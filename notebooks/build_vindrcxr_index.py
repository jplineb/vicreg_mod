import sys
sys.path.insert(0,"..")
from custom_datasets import VINDR_CXR_BASE

data_path = "/project/dane2/wficai/BenchMD/vindr-cxr/1.0.0/"

dataset = VINDR_CXR_BASE(base_root=data_path, build_index=True)

# dataset = VINDR_CXR_BASE(base_root=data_path, train=False)
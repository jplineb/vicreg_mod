import torch
import sys

# Usage: python inspect_checkpoint.py path/to/checkpoint.pth

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location='cpu')

# If the checkpoint is a dict with 'model' or 'state_dict', extract it
if isinstance(ckpt, dict):
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

for name, param in state_dict.items():
    if hasattr(param, 'dtype'):
        print(f"{name}: dtype={param.dtype}, shape={param.shape}, min={param.min().item()}, max={param.max().item()}")
    else:
        print(f"{name}: type={type(param)}")

import torch
from torch import nn
from torch.utils.data import DataLoader  # wraps an iterable around the Dataset
from torchvision import datasets  # stores the samples and their corresponding labels
from torchvision.transforms import ToTensor  # visual dataset
import time
import sys
import netron
from IPython.display import IFrame
from brevitas.export import export_qonnx # for exporting ONNX
from CNV import cnv
from trainer import Trainer, EarlyStopper
from reporting import *

project_name = "CNV"
weight_bit_width = 1
act_bit_width = 2

formatted_PTH_file_size = "{:.2f}".format(os.path.getsize(f"model_{project_name}_W{weight_bit_width}A{act_bit_width}.pth") / (1024 * 1024))

report = f" PTH File Size: {formatted_PTH_file_size} MB"

print(formatted_PTH_file_size)

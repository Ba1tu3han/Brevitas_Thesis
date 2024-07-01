
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

weight_bit_width = 1
act_bit_width = 1
in_bit_width = 8
num_classes = 10
n_channel = 3

#  DEVICE CHECK FOR TRAINING
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")
if device == 'cpu':
    sys.exit("It is stopped because device is selected as CPU")

model = cnv(n_channel, weight_bit_width, act_bit_width, in_bit_width, num_classes).to(device)
model.load_state_dict(torch.load("model.pth"))


# Load the model weights from the file
model_path = 'model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully from:", model_path)
else:
    print("Error: Model file not found at:", model_path)
    exit(1)

# Print the number of layers
num_layers = len(list(model.parameters()))
print("Number of layers:", num_layers)

# Print the number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:", num_params)

# Print the file size of the model
file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to megabytes
print("File size of the model:", "{:.2f} MB".format(file_size))
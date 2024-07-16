# LIBRARIES
import torch
from torch import nn
from torch.utils.data import DataLoader  # wraps an iterable around the Dataset
from torchvision import datasets  # stores the samples and their corresponding labels
from torchvision.transforms import ToTensor, Resize, Compose  # visual dataset
import time
import sys
import netron
from IPython.display import IFrame
from brevitas.export import export_qonnx # for exporting ONNX

from CNV import cnv
from trainer import Trainer, EarlyStopper
from reporting import *


resize_tensor = Compose([
  Resize([32, 32]), # target size of dataset images
  ToTensor()
])


# PROCESS TIME MEASURING
process_start_time = time.time() # to measure whole processing time
print("Name of this attempt is: " + str(process_start_time))


#  DOWNLOAD TRAINING AND TEST DATASETS FROM OPEN DATASETS
batch_size = 32
n_sample = None

training_data = datasets.GTSRB(
    root = "data",
    split = "train",
    download=True,
    transform=resize_tensor
)

test_data = datasets.GTSRB(
    root = "data",
    split = "test",
    download = True,
    transform=resize_tensor
)


#  CREATE DATA LOADERS
train_dataloader = DataLoader(training_data, batch_size=batch_size,
                              sampler=torch.utils.data.RandomSampler(training_data, num_samples=n_sample))
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                             sampler=torch.utils.data.RandomSampler(test_data, num_samples=n_sample))

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# IMAGE INFO
N, n_channel, shape_y, shape_x = X.shape
print('data info', N, n_channel, shape_y, shape_x)

#  RESHAPE IMAGES
# use opencv


#  DEVICE CHECK FOR TRAINING
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")
if device == 'cpu':
    sys.exit("It is stopped because device is selected as CPU")

#QUANTIZATION CONFIGURATION
weight_bit_width = 1
act_bit_width = 1
in_bit_width = 8
num_classes = 43

#  DEFINING A MODEL
config = "skip"

model = cnv(n_channel, weight_bit_width, act_bit_width, in_bit_width, num_classes)
model = model.to(device)  # moving the model to the device
model_name = {type(model).__name__}
model_name = model_name.pop()


# LOADING THE TRAINED MODEL
model = cnv(n_channel, weight_bit_width, act_bit_width, in_bit_width, num_classes).to(device)
model.load_state_dict(torch.load("model.pth"))


# EVALUATING THE MODEL
classes = list(range(num_classes))

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.reshape((1,)+tuple(x.shape)).to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

import numpy as np

out_img2=test_data[0][0].cpu().detach().numpy()
arr_ = np.squeeze(out_img2)
plt.imshow(arr_)
plt.show()


process_end_time = time.time() # Printing the processing time
time_diff = process_end_time - process_start_time
print(f"Process Time [min]: {time_diff / 60:.2f}")

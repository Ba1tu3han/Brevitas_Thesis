# LIBRARIES
import os
import onnx
import torch
from torch import nn
from torchvision.transforms import ToTensor, Resize, Compose  # visual dataset
import time
import sys
import netron
from IPython.display import IFrame
import numpy as np

from trainer import Trainer, EarlyStopper
from reporting import *

# MEASURING PROCESS TIME

process_start_time = time.time() # to measure whole processing time
print("Code of this attempt is: " + str(process_start_time))

#  DEVICE CHECK

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")
if device == 'cpu':
    sys.exit("It is stopped because device is selected as CPU")

# REPRODUCTIVITY

torch.manual_seed(0) # for torch, Setting seeds for reproducibility, keeps random numbers the same
np.random.seed(0) # for numpy
torch.cuda.manual_seed(0) #for reproducibility
torch.backends.cudnn.deterministic = True #for reproducibility
torch.backends.cudnn.benchmark = False #for reproducibility


#  DOWNLOAD TRAINING AND TEST DATASETS FROM OPEN DATASETS
from torchvision import datasets  # stores the samples and their corresponding labels


batch_size = 32
n_sample = None

resize_tensor = Compose([ # resizing dataset images
  Resize([32, 32]), # target size of dataset images # 32x32 is for the original CNV network for GTSRB
  ToTensor()
])

train_validation_data = datasets.GTSRB( # German Traffic Sign Dataset
    root = "data",
    split = "train",
    download=True,
    transform=resize_tensor
)
train_size = int(0.8 * len(train_validation_data))
validation_size = len(train_validation_data) - train_size
training_data, validation_data = torch.utils.data.random_split(train_validation_data, [train_size, validation_size])

test_data = datasets.GTSRB(
    root = "data",
    split = "test",
    download = True,
    transform=resize_tensor
)
print("DOWNLOAD TRAINING AND TEST DATASETS FROM OPEN DATASETS is done")


#  SETTINGS UP DATALOADERS
from torch.utils.data import DataLoader  # wraps an iterable around the Dataset

train_dataloader = DataLoader(training_data, batch_size=batch_size,
                              sampler=torch.utils.data.RandomSampler(training_data, num_samples=n_sample))
validation_dataloader = DataLoader(validation_data, batch_size=batch_size,
                                   sampler=torch.utils.data.RandomSampler(validation_data, num_samples=n_sample))
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                             sampler=torch.utils.data.RandomSampler(test_data, num_samples=n_sample))

print(f"Train size: {len(train_dataloader)}\n"
      f"Validation size: {len(validation_dataloader)}\n"
      f"Test Size: {len(test_dataloader)}")
# for train: shuffle=True for test: shuffle:False can be added

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# IMAGE INFO
N, n_channel, shape_y, shape_x = X.shape
print('data info', N, n_channel, shape_y, shape_x)

print("SETTINGS UP DATALOADERS is done")

# DEFINING A MODEL

#from CNV import cnv # Original CNV network. Be careful "import cnv" shall be lower case.
from CNV_light import cnv # light version of the CNV
project_name = "CNV_light" # to name the output onnx file. "CNV" or "CNV_light"

weight_bit_width = 5 # quantization configuration for weights
act_bit_width = 5 # quantization configuration for activation functions
in_bit_width = 8 # bit width of input
num_classes = 43 # number of class

config = "skip"

model = cnv(n_channel, weight_bit_width, act_bit_width, in_bit_width, num_classes)
model = model.to(device)  # moving the model to the device
model_name = {type(model).__name__}
model_name = model_name.pop()

print("DEFINING A MODEL is done")

# TRAINING

# from losses import SqrHingeLoss
# loss_fn = SqrHingeLoss()  # loss function

loss_fn = nn.CrossEntropyLoss()  # loss function
lr = 1e-3 # the best practice is 4e-3
epochs = 2 # upper limit of the number of epoch
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    train_dataloader=train_dataloader,
    val_dataloader=validation_dataloader,
    test_dataloader=test_dataloader,
    sample_size=n_sample
)
train_accuracies = []
validation_accuracies = []
train_f1s = []
validation_f1s = []
train_losses = []
validation_losses = []


min_delta = 0
patience = 1502 # best practice is 15
early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
early_stopper_flag = False # for the Brevitas Report
from tqdm import tqdm
for t in tqdm(range(epochs)):
    print(f"Epoch {t + 1}\n-------------------------------")
    epoch_train_accuracy, epoch_train_f1, epoch_train_loss = trainer.train_one_epoch()
    epoch_validation_accuracy, epoch_validation_f1, epoch_validation_loss = trainer.validate_one_epoch()

    train_accuracies.append(epoch_train_accuracy)
    validation_accuracies.append(epoch_validation_accuracy)

    train_f1s.append(epoch_train_f1)
    validation_f1s.append(epoch_validation_f1)

    train_losses.append(epoch_train_loss)
    validation_losses.append(epoch_validation_loss)

    if early_stopper.early_stop(epoch_validation_loss):
        print(f"The training loop is stopped due to no improvement in accuracy. "
              f"It is equal to or smaller than {min_delta}")
        print(f"Number of epoch: {t}")
        early_stopper_flag = True # for the Brevitas Report
        break

print("TRAINING is Done!")

run_info = f"{project_name}_W{weight_bit_width}A{act_bit_width}"

#  SAVING THE MODEL

torch.save(model.state_dict(), f"model_{run_info}.pth")
print(f"SAVING THE MODEL is done. File name is model_{run_info}.pth")


# EXPORTING AND CLEANING UP QONNX

from brevitas.export import export_qonnx # for exporting ONNX
from qonnx.util.cleanup import cleanup as qonnx_cleanup # pip install qonnx

input_tensor = torch.randn(1, n_channel, shape_x, shape_y).to(device) # bach size must be 1 https://github.com/Xilinx/finn/discussions/1029
export_path = f"QONNX_{run_info}.onnx"
export_qonnx(model, export_path=export_path, input_t=input_tensor)
qonnx_cleanup(export_path, out_file=export_path)

print("EXPORTING AND CLEANING UP QONNX is Done!")


# MODEL VISUALIZATION
def show_netron(model_path, port):
    time.sleep(3.)
    netron.start(model_path, address=("localhost", port), browse=False)
    return IFrame(src=f"http://localhost:{port}/", width="100%", height=400)
show_netron(export_path, 8082) # not necessary

print("MODEL VISUALIZATION is Done!")


# EVALUATING THE MODEL

model = cnv(n_channel, weight_bit_width, act_bit_width, in_bit_width, num_classes).to(device) # LOADING THE TRAINED MODEL
model.load_state_dict(torch.load(f"model_{run_info}.pth"))

classes = list(range(num_classes))


report_name = f"Classification_Report_{run_info} {process_start_time}.txt"
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.reshape((1,)+tuple(x.shape)).to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    test_accuracy, test_precision, test_recall, test_f1, test_loss = trainer.test(report_name)
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

process_end_time = time.time() # Printing the processing time
time_diff = process_end_time - process_start_time
print(f"Process Time [min]: {time_diff / 60:.2f}")
print("EVALUATING THE MODEL is Done!")


# REPORT

formatted_ONNX_file_size = "{:.2f}".format(os.path.getsize(export_path) / (1024 * 1024))
formatted_PTH_file_size = "{:.2f}".format(os.path.getsize(f"model_{run_info}.pth") / (1024 * 1024))

from torchinfo import summary

model_stats = summary(model, input_size=(batch_size, n_channel, shape_y, shape_x))

total_size = sum(p.numel() * p.element_size() for p in model.parameters()) # for memory usage


report = f"""\
Validation Accuracy: {epoch_validation_accuracy :.4f}
Validation F1 (macro): {epoch_validation_f1 :.4f}
Validation Loss: {epoch_validation_loss :.4f}%

Test Loss: {test_loss :.4f}%
Test Accuracy: {test_accuracy :.4f}
Test Precision (macro) : {test_precision :.4f}
Test Recall (macro) : {test_recall :.4f}
Test F1 (macro) : {test_f1 :.4f}


Export Path: {export_path}
Process Time [min]: {time_diff / 60:.2f}

----------------------------------

Image Channel: {n_channel}
Image Size: {shape_y} x {shape_x}
Batch Size: {batch_size}
Number of Class: {num_classes}

Project Name: {project_name}
Number of Layer: {len(list(model.parameters()))}
Number of Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}
Memory Usage: {total_size / (1024 ** 2):.2f} MB

Number of Upper Limit Epoch: {epochs}
Number of Epoch {t}
Optimizer: {type(optimizer).__name__}
Learning Rate: {lr}
Early Stopper: {early_stopper_flag}
Early Stopper Min Delta: {min_delta}
Early Stopper Patience: {patience}

Quantization: W{weight_bit_width}A{act_bit_width}
Input Bit Width: {in_bit_width}

Device = {device}
ONNX File Size: {formatted_ONNX_file_size} MB
PTH File Size: {formatted_PTH_file_size} MB

{model_stats}
"""
#Dataset: {training_data.file}
export_brevitas_report(report, process_start_time)
print("REPORTING is Done!")

# PLOTTING ACCURACY GRAPH

export_accuracy_graph(train_losses, validation_losses,
                      train_accuracies, validation_accuracies,
                      train_f1s, validation_f1s,
                      process_start_time, run_info) # PLOTTING ACCURACY GRAPH




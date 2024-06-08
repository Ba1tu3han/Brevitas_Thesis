# LIBRARIES
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

# PROCESS TIME MEASURING
process_start_time = time.time() # to measure whole processing time
print("Name of this attempt is: " + str(process_start_time))


#  DOWNLOAD TRAINING AND TEST DATASETS FROM OPEN DATASETS
batch_size = 32
n_sample = None

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
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
weight_bit_width = 2
act_bit_width = 2
in_bit_width = 8
num_classes = 10

#  DEFINING A MODEL
config = "skip"

model = cnv(n_channel, weight_bit_width, act_bit_width, in_bit_width, num_classes)
model = model.to(device)  # moving the model to the device


# TRAINING

# from losses import SqrHingeLoss
# loss_fn = SqrHingeLoss()  # loss function

loss_fn = nn.CrossEntropyLoss()  # loss function
lr = 4e-3
epochs = 100 # upper limit of number of epoch
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    test_dataloader=test_dataloader,
    sample_size=n_sample
)
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

min_delta = 0.001
patience = 3
early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
early_stopper_flag = False # for the Brevitas Report

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    epoch_train_accuracy, epoch_train_loss = trainer.train_one_epoch()
    epoch_test_accuracy, epoch_test_loss = trainer.validate_one_epoch()

    train_accuracies.append(epoch_train_accuracy)
    test_accuracies.append(epoch_test_accuracy)
    train_losses.append(epoch_train_loss)
    test_losses.append(epoch_test_loss)

    if early_stopper.early_stop(epoch_test_loss):
        print(f"The training loop is stopped due to no improvement in accuracy. "
              f"It is equal to or smaller than {min_delta}")
        print(f"Number of epoch: {t}")
        early_stopper_flag = True # for the Brevitas Report
        break

print("Training is Done!")

#  SAVING THE MODEL
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# EXPORT QONNX
input_tensor = torch.randn(1, n_channel, shape_x, shape_y).to(device)
export_path = f"QONNX_CNV_{weight_bit_width}W{act_bit_width}A.onnx"
export_qonnx(model, input_tensor, export_path=export_path)


# MODEL VISUALIZATION
def show_netron(model_path, port):
    time.sleep(3.)
    netron.start(model_path, address=("localhost", port), browse=False)
    return IFrame(src=f"http://localhost:{port}/", width="100%", height=400)
show_netron("./QONNX_CNV.onnx", 8082)

# LOADING THE TRAINED MODEL
model = cnv(n_channel, weight_bit_width, act_bit_width, in_bit_width, num_classes).to(device)
model.load_state_dict(torch.load("model.pth"))


# EVALUATING THE MODEL
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.reshape((1,)+tuple(x.shape)).to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

process_end_time = time.time() # Printing the processing time
time_diff = process_end_time - process_start_time
print(f"Process Time [min]: {time_diff / 60:.2f}")

# REPORT
formatted_file_size = "{:.2f}".format(os.path.getsize(export_path) / (1024 * 1024)) # for ONNX File Size

report = f"""Validation Accuracy: {epoch_test_accuracy :.4f}
Validation Loss: {epoch_test_loss :.4f}%

Export Path: {export_path}
Process Time [min]: {time_diff / 60:.2f}

----------------------------------

Dataset: {training_data.filename}
Image Channel: {n_channel}
Image Size: {shape_y} x {shape_x}
Batch Size: {batch_size}
Number of Class: {num_classes}

Model: {type(model).__name__}
Number of Layer: {len(list(model.parameters()))}
Number of Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}

Number of Upper Limit Epoch: {epochs}
Number of Epoch {t}
Optimizer: {type(optimizer).__name__}
Learning Rate: {lr}
Early Stopper: {early_stopper_flag}
Early Stopper Min Delta: {min_delta}
Early Stopper Patience: {patience}

Quantization: {weight_bit_width}W{act_bit_width}A
Input Bit Width: {in_bit_width}

Device = {device}
ONNX File Size: {formatted_file_size} MB
"""

export_brevitas_report(report, process_start_time)

#export_accuracy_graph(train_losses, test_losses, train_accuracies, test_accuracies, process_start_time, epochs-1) # PLOTTING ACCURACY GRAPH
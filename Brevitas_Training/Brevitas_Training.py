#  SOURCE: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

#   SUMMARY
# Model:            Brevitas CNV
# Dataset:          CIFAR10
# Quantization:     QAT
# Export Format:    QONNX
# Bit Width:        W1A1

#  LIBRARIES

import torch
from torch import nn
from torch.utils.data import DataLoader  # wraps an iterable around the Dataset
from torchvision import datasets  # stores the samples and their corresponding labels
from torchvision.transforms import Compose, Resize, ToTensor  # visual dataset

import time

from torch.nn import Module
import torch.nn.functional as F
import brevitas.nn as qnn


process_start_time = time.time()

#  DOWNLOAD TRAINING AND TEST DATASETS FROM OPEN DATASETS

batch_size = 32
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

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

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
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#  DEFINING A MODEL

from CNV import cnv

config = "skip"
model = cnv(n_channel=n_channel)

model = model.to(device)  # moving the model to the device
# print(model)

#  OPTIMIZING THE MODEL PARAMETERS

loss_fn = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3)  # optimizer
# Plot it and decide the learning rate
# learning rate finder for pytorch

def train(dataloader, model, loss_fn, optimizer):  # training function
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):  # testing function
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy (Top1): {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, test_loss


# TRAINING

epochs = 30 # upper limit of number of epoch

epoch_accuracies = [] # lists to store accuracy and loss for each epoch
epoch_losses = []
stop_delta_counter = 0 # count the number of accuracy_delta which is less than stop_delta

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy_list, loss_list = test(test_dataloader, model, loss_fn)

    epoch_accuracies.append(accuracy_list) # storing all accuracies
    epoch_losses.append(loss_list) # storing all losses

    accuracy_delta = epoch_accuracies[t] - epoch_accuracies[t - 1] # current pace (change) of accuracy
    loss_delta = epoch_losses[t] - epoch_losses[t - 1]  # current pace (change) of loss
    #print("epoch_accuracies: " + str(epoch_accuracies))
    #print("epoch_losses: " + str(epoch_losses))
    print("accuracy_delta: " + str(accuracy_delta))
    #print("loss_delta: " + str(loss_delta))

    stop_delta = 0.2
    if abs(accuracy_delta) < stop_delta and t > 2: # stopping the training loop due to no improvement in accuracy. First 2 steps are ignored.

        stop_delta_counter += 1
        if stop_delta_counter == 2: # two times accuracy_delta is less than stop_delta
            print("The training loop is stopped due to no improvement in accuracy. It is smaller than " + str(stop_delta))
            print("Number of epoch: " + str(t))
            break

print("Training is Done!")


#  SAVING THE MODEL

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# EXPORT QONNX
# opset version must be compatible with FINN compiler. Therefor, it is 17.

from brevitas.export import export_qonnx

#input_tensor = torch.randn(batch_size, n_channel, shape_x, shape_y).to(device)
input_tensor = torch.randn(1, n_channel, shape_x, shape_y).to(device)
export_qonnx(model, input_tensor, export_path='QONNX_CNV.onnx')


# VISUALIZATION

import netron
import time
from IPython.display import IFrame
def show_netron(model_path, port):
    time.sleep(3.)
    netron.start(model_path, address=("localhost", port), browse=False)
    return IFrame(src=f"http://localhost:{port}/", width="100%", height=400)

show_netron("./QONNX_CNV.onnx", 8082)


#  LOADING A MODEL

model = cnv(n_channel=n_channel).to(device)
model.load_state_dict(torch.load("model.pth"))


#  EVALUATING THE MODEL

classes = [ #  can be deleted
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

process_end_time = time.time()
time_diff = process_end_time - process_start_time
print("Process Time [min]: " + str(time_diff/60))
#print("Number of Epoch: " + str(epochs))



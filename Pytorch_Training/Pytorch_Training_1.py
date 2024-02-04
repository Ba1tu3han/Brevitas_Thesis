#  SOURCE: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

#  LIBRARIES

import torch
from torch import nn
from torch.utils.data import DataLoader  # wraps an iterable around the Dataset
from torchvision import datasets  # stores the samples and their corresponding labels
from torchvision.transforms import ToTensor  # visual dataset

from torch.nn import Module
import torch.nn.functional as F
import brevitas.nn as qnn

#  DOWNLOAD TRAINING DATA FROM OPEN DATASETS

batch_size = 64
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

#  DOWNLOAD TEST DATA FROM OPEN DATASETS

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#  CREATE DATA LOADERS

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#  DEVICE CHECK FOR TRAINING

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
----------------------------------------------------------------------------------------------------
#  DEFINING A MODEL

from mobilenetv1 import quant_mobilenet_v1

#  Bit-Width Configuration

config = {
    'QUANT': {
        'WEIGHT_BIT_WIDTH': 4,
        'ACT_BIT_WIDTH': 4,
        'IN_BIT_WIDTH': 4,
    },
    'MODEL': {
        'NUM_CLASSES': 10,
        'IN_CHANNELS': 1,  # Adjusted to match MNIST dataset
    }
}

# Instantiate the FC model with extracted parameters
model = quant_mobilenet_v1(config)

model = model().to(device)  # moving the model to the device
print(model)
--------------------------------------------------------------------------------
#  OPTIMIZING THE MODEL PARAMETERS

loss_fn = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # optimizer


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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# TRAINING

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#  SAVING THE MODEL

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#  LOADING A MODEL

model = quant_mobilenet_v1(cfg).to(device)
model.load_state_dict(torch.load("model.pth"))

#  EVALUATING THE MODEL

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

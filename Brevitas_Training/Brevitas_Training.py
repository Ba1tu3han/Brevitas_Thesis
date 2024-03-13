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
from torchvision.transforms import ToTensor  # visual dataset

import time
import matplotlib.pyplot as plt  # to plot graphs

from losses import SqrHingeLoss
from trainer import Trainer, EarlyStopper

process_start_time = time.time()
print("Name of this attempt is: " + str(process_start_time))


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

n_sample = None

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
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


#  DEFINING A MODEL

from CNV import cnv

config = "skip"
model = cnv(n_channel=n_channel)

model = model.to(device)  # moving the model to the device
# print(model)

# TRAINING

#  OPTIMIZING THE MODEL PARAMETERS

# learning rate finder for pytorch

# from losses import SqrHingeLoss
# loss_fn = SqrHingeLoss()  # loss function
loss_fn = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)  # optimizer
epochs = 1100  # upper limit of number of epoch
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

min_delta = 0.01
early_stopper = EarlyStopper(patience=3,
                             min_delta=min_delta)
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
        break

print("Training is Done!")

# Plotting Accuracy Graph
epoch_list = list(range(t + 1))  # to list epochs 1 to the end
fig, ax = plt.subplots(2, 1)

ax[0].plot(epoch_list, train_losses, label="Train Loss")
ax[0].plot(epoch_list, test_losses, label="Validation Loss")
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(epoch_list, train_accuracies, label="Train Accuracy")
ax[1].plot(epoch_list, test_accuracies, label="Validation Accuracy")
ax[1].set_xlabel('Number of Epoch')
ax[1].set_ylabel('Accuracy (Top1)')
ax[1].legend()

figure_name = "Accuracy_Loss_Plot " + str(process_start_time) + ".png"
plt.savefig(figure_name)


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

# ONNX Flow-Chart
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
print(f"Process Time [min]: {time_diff / 60:.2f}")
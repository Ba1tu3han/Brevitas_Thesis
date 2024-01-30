# 1) Libraries

from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import brevitas.nn as qnn
from brevitas.quant import Int32Bias


# 2) Dataset - MNIST

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  # normalization for MNIST
])

# Training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 3) Neural Network Model       

# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import ast
from functools import reduce
from operator import mul

import torch
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear

from common import CommonActQuant
from common import CommonWeightQuant
from tensor_norm import TensorNorm

DROPOUT = 0.2

#https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/bnn_pynq/models/FC.py

class FC(Module):

    def __init__(
        self,
        num_classes,
        weight_bit_width,
        act_bit_width,
        in_bit_width,
        in_channels,
        out_features,
        in_features=(28, 28)):
        super(FC, self).__init__()

        self.features = ModuleList()
        self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width))
        self.features.append(Dropout(p=DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in out_features:
            self.features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_bit_width=weight_bit_width,
                    weight_quant=CommonWeightQuant))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            self.features.append(Dropout(p=DROPOUT))
        self.features.append(
            QuantLinear(
                in_features=in_features,
                out_features=num_classes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant))
        self.features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.features:
            x = mod(x)
        return x


def fc(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    out_features = ast.literal_eval(cfg.get('MODEL', 'OUT_FEATURES'))
    net = FC(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        in_channels=in_channels,
        out_features=out_features,
        num_classes=num_classes)
    return net
    
# Configuration as a dictionary
# https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/bnn_pynq/cfg

config = {
    'QUANT': {
        'WEIGHT_BIT_WIDTH': 1,
        'ACT_BIT_WIDTH': 1,
        'IN_BIT_WIDTH': 1,
    },
    'MODEL': {
        'NUM_CLASSES': 10,
        'IN_CHANNELS': 1, # Adjusted to match MNIST dataset
    }
}

# Extract parameters from the config dictionary
num_classes = config['MODEL']['NUM_CLASSES']
weight_bit_width = config['QUANT']['WEIGHT_BIT_WIDTH']
act_bit_width = config['QUANT']['ACT_BIT_WIDTH']
in_bit_width = config['QUANT']['IN_BIT_WIDTH']
in_channels = config['MODEL']['IN_CHANNELS']

# Instantiate the FC model with extracted parameters
quant_weight_act_bias_lenet = FC(num_classes=num_classes,
                                 weight_bit_width=weight_bit_width,
                                 act_bit_width=act_bit_width,
                                 in_bit_width=in_bit_width,
                                 in_channels=in_channels,
                                 out_features=[64, 64, 64])



# 4) Training the Network Model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device select,on GPU or CPU
print(device)

quant_weight_act_bias_lenet = quant_weight_act_bias_lenet.to(device) # moving the model to selected device

criterion = nn.CrossEntropyLoss() # loss function and optimizer
optimizer = optim.Adam(quant_weight_act_bias_lenet.parameters(), lr=0.001)

# Traning Loop
num_epochs = 8
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = quant_weight_act_bias_lenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() # Printing results
        if i % 200 == 199:  # mini batch
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Training is done')

# 5) Saving 

PATH = './QONNX_FC.pth'
torch.save(quant_weight_act_bias_lenet.state_dict(), PATH)

# EXPORT QONNX 

#from brevitas.export import export_onnx_qcdq
from brevitas.export import export_qonnx

input_tensor = torch.randn(1, 1, 28, 28).to(device) # size for MNIST
export_qonnx(quant_weight_act_bias_lenet, input_tensor, export_path='QONNX_FC.onnx')

# 6) Validation on Test Dataset

quant_weight_act_bias_lenet.eval()  # Evaluation mode

correct = 0
total = 0
with torch.no_grad():  # gradient calculation
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = quant_weight_act_bias_lenet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test %d %%' % (100 * correct / total))


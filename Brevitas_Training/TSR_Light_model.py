# Paper Source of Network: A Lightweight Convolutional Neural Network (CNN) Architecture for Trafﬁc Sign Recognition in Urban Road Networks
# Brevitas Source: https://github.com/Xilinx/finn/blob/e3087ad9fbabcc35f21164d415ababec4f462e9f/notebooks/end2end_example/cybersecurity/1-train-mlp-with-brevitas.ipynb

# THIS IS NOT WORKING YET

import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d, QuantLinear, QuantIdentity


class TSRNet(nn.Module):
    def __init__(self, weight_bit_width, act_bit_width):
        super(TSRNet, self).__init__()

        self.conv1 = QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1, weight_bit_width=weight_bit_width)
        self.relu1 = QuantReLU(bit_width=act_bit_width)
        self.pool1 = QuantMaxPool2d(kernel_size=2, stride=2)

        self.conv2 = QuantConv2d(16, 32, kernel_size=3, stride=1, padding=1, weight_bit_width=weight_bit_width)
        self.relu2 = QuantReLU(bit_width=act_bit_width)
        self.pool2 = QuantMaxPool2d(kernel_size=2, stride=2)

        self.conv3 = QuantConv2d(32, 64, kernel_size=3, stride=1, padding=1, weight_bit_width=weight_bit_width)
        self.relu3 = QuantReLU(bit_width=act_bit_width)
        self.pool3 = QuantMaxPool2d(kernel_size=2, stride=2)

        self.conv4 = QuantConv2d(64, 128, kernel_size=3, stride=1, padding=1, weight_bit_width=weight_bit_width)
        self.relu4 = QuantReLU(bit_width=act_bit_width)
        self.pool4 = QuantMaxPool2d(kernel_size=2, stride=2)

        self.conv5 = QuantConv2d(128, 256, kernel_size=3, stride=1, padding=1, weight_bit_width=weight_bit_width)
        self.relu5 = QuantReLU(bit_width=act_bit_width)
        self.pool5 = QuantMaxPool2d(kernel_size=2, stride=2)

        self.fc1 = QuantLinear(256 * 1 * 1, 512, weight_bit_width=weight_bit_width)
        self.relu_fc1 = QuantReLU(bit_width=act_bit_width)

        self.fc2 = QuantLinear(512, 43, weight_bit_width=weight_bit_width)
        self.quant_output = QuantIdentity(bit_width=act_bit_width)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        x = self.quant_output(x)
        x = F.softmax(x, dim=1)
        return x


# Example usage:
weight_bit_width = 4
act_bit_width = 4

model = TSRNet(weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
print(model)

# Image Classification Model for GTSRB Dataset

input_size = 32
hidden1 = 64
hidden2 = 64
hidden3 = 64
weight_bit_width = 2
act_bit_width = 2
num_classes = 43


from brevitas.nn import QuantLinear, QuantReLU
import torch.nn as nn

# Setting seeds for reproducibility
torch.manual_seed(0)

model = nn.Sequential(

      QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bit_width),
      nn.BatchNorm1d(hidden1),
      nn.Dropout(0.5),
      QuantReLU(bit_width=act_bit_width),

      QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bit_width),
      nn.BatchNorm1d(hidden2),
      nn.Dropout(0.5),
      QuantReLU(bit_width=act_bit_width),

      QuantLinear(hidden2, hidden3, bias=True, weight_bit_width=weight_bit_width),
      nn.BatchNorm1d(hidden3),
      nn.Dropout(0.5),
      QuantReLU(bit_width=act_bit_width),

      QuantLinear(hidden3, num_classes, bias=True, weight_bit_width=weight_bit_width)
)
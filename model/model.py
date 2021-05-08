import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


def find_previous_layer(i, layers):
    if i == 0:
        return None

    for j in range(i - 1, -1, -1):
        if type(layers[j]) is int:
            return layers[j]


class HomeTaskModel(BaseModel):
    def __init__(self, layers_list):
        super(HomeTaskModel, self).__init__()

        layers = []
        for i, layer in enumerate(layers_list):
            previous_layer = find_previous_layer(i, layers_list)
            if previous_layer is None:
                continue

            if layer == "sigmoid":
                current_layer = nn.Sigmoid()
            elif layer == "relu":
                current_layer = nn.ReLU()
            elif layer == "tanh":
                current_layer = nn.Tanh()
            else:
                current_layer = nn.Linear(previous_layer, layer)

            layers.append(current_layer)

        layers.append(torch.nn.Softmax())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

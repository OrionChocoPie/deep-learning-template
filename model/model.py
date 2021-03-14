import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class HomeTaskModel(BaseModel):
    def __init__(self, layers_list):
        super(HomeTaskModel, self).__init__()
        
        layers = []
        for layer in layers_list:
            layers.append(torch.nn.Linear(layer[0], layer[1]))

            if (layer[2] == 'sigmoid'):
                layers.append(torch.nn.Sigmoid())
            elif (layer[2] == 'relu'):
                layers.append(torch.nn.ReLU())
            elif (layer[2] == 'tanh'):
                layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Softmax())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
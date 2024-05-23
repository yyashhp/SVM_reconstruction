import numpy as np
import torch
import pandas as pd

from torch import nn

class Binary_SVM(nn.Module):

    def __init__(self, n_features):
        super(Binary_SVM, self).__init__()
        self.linear = nn.Linear(n_features, 1) #For SVM we have one linear layer, input size n, output size 1


    def forward(self, x): #x is input
        out = self.linear(x) #returns tensor of size (N x 1)
        out = out.squeeze() #(N x 1) -> N
        return out
    

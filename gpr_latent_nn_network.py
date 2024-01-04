import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpNet(nn.Module):
    def __init__(self, input_dim):
        super(InterpNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(2 * input_dim + 1, input_dim),  # Input is two vectors + label value
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),  # Output is one vector
            # nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.seq(x)

        return x

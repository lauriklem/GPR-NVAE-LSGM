import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpNetConv(nn.Module):
    def __init__(self, input_dim, act_fn=nn.LeakyReLU()):
        super(InterpNetConv, self).__init__()
        c_in, h_in, w_in = input_dim
        c_out = int(c_in / 3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, 32 * c_in, kernel_size=3, padding=1, groups=c_out),
            act_fn,
            nn.Dropout(0.2),
            nn.Conv2d(32 * c_in, 32 * c_in, kernel_size=3, padding=1, groups=c_out),
            act_fn,
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Conv2d(32 * c_in, c_out, kernel_size=1, groups=c_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


"""
self.seq = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=1, groups=c_out),
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(c_in, c_out, kernel_size=1, groups=c_out),
            )
"""


"""
            act_fn,
            nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1),  # 41 x 32 x 32
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1),  # 41 x 32 x 32
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(in_channels=c_in, out_channels=2 * c_in, kernel_size=3, padding=1, stride=2),  # 82 x 16 x 16
            nn.BatchNorm2d(2 * c_in),
            act_fn,
            nn.Conv2d(in_channels=2 * c_in, out_channels=2 * c_in, kernel_size=3, padding=1),  # 82 x 16 x 16
            nn.BatchNorm2d(2 * c_in),
            act_fn,

            nn.ConvTranspose2d(in_channels=2 * c_in, out_channels=2 * c_in, kernel_size=4, stride=2, padding=1),  # 82 x 32 x 32
            nn.BatchNorm2d(2 * c_in),
            act_fn,
            nn.Conv2d(in_channels=2 * c_in, out_channels=c_in, kernel_size=1),  # 41 x 32 x 32
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1),  # 41 x 32 x 32
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(c_in, c_out, kernel_size=1),  # 20 x 32 x 32
            nn.Tanh()
            """

"""
            nn.Conv2d(c_in, c_in, kernel_size=1, groups=c_out),
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(c_in, c_in, kernel_size=1, groups=c_out),
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(c_in, c_out, kernel_size=1, groups=c_out),
            nn.BatchNorm2d(c_out),
            act_fn,
            nn.Conv2d(c_out, c_out, kernel_size=1, groups=c_out),
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpNetConv(nn.Module):
    def __init__(self, input_dim, n_latents, act_fn=nn.ReLU()):
        super(InterpNetConv, self).__init__()
        self.seq_list = nn.ModuleList()
        c_in, h_in, w_in = input_dim
        c_out = int((c_in - 1) / 2)
        # c_out = int(c_in / 3)
        self.n_latents = n_latents

        for n in range(n_latents):
            self.seq_list.append(nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1),  # 20 x 32 x 32
            ))

    def forward(self, x):
        y = []
        for i in range(self.n_latents):  # batch, n_latents, channels, width, height
            y.append(self.seq_list[i](x[:, i, :, :, :]))

        return torch.stack(y)


"""
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=2, padding=1),  # 40 x 32 x 32
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=2, padding=1),  # 40 x 16 x 16
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            act_fn,

            nn.ConvTranspose2d(in_channels=c_out, out_channels=c_out, kernel_size=4, padding=1, stride=2),  # 40 x 32 x 32
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(in_channels=c_out, out_channels=c_out, kernel_size=4, padding=1, stride=2),  # 40 x 64 x 64
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            """

"""
nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1),  # 81 x 64 x 64
            act_fn,
            nn.BatchNorm2d(c_in),

            nn.Conv2d(in_channels=c_in, out_channels=int((c_in + c_out) / 2), kernel_size=1),  # 60 x 64 x 64
            act_fn,
            nn.BatchNorm2d(int((c_in + c_out) / 2)),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=int((c_in + c_out) / 2), out_channels=int((c_in + c_out) / 2), kernel_size=3, padding=1),  # 60 x 64 x 64
            act_fn,
            nn.BatchNorm2d(int((c_in + c_out) / 2)),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=int((c_in + c_out) / 2), out_channels=c_out, kernel_size=1),  # 40 x 64 x 64
            nn.Tanh()
            """


"""
nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, groups=c_out),
            act_fn,
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
"""
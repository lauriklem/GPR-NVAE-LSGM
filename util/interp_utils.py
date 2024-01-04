import torch
import numpy as np


def linear_interpolation(tensor1, tensor2, c=0.5):
    return ((1 - c) * tensor1 + c * tensor2).unsqueeze(0)


def slerp2(tensor1, tensor2, c):
    inner_product = (tensor1 * tensor2).sum(dim=0)
    t1_norm = tensor1.pow(2).sum(dim=0).pow(0.5)
    t2_norm = tensor2.pow(2).sum(dim=0).pow(0.5)

    cos = inner_product / (t1_norm * t2_norm)
    cos = torch.clamp(cos, -1, 1)

    angle = torch.acos(cos)

    s_angle = torch.sin(angle)  # sine of angle
    s_angle += (s_angle == 0) * torch.finfo(torch.float32).eps  # for numerical stability

    interpolated = (torch.sin((1.0 - c) * angle) / s_angle * tensor1 + torch.sin(c * angle) / s_angle * tensor2)

    return interpolated.unsqueeze(0)


def slerp(tensor1, tensor2, c):
    t1_norm = tensor1 / torch.norm(tensor1, dim=0, keepdim=True)
    t2_norm = tensor2 / torch.norm(tensor2, dim=0, keepdim=True)
    cos = (t1_norm * t2_norm).sum(0)
    cos = torch.clamp(cos, -1, 1)

    angle = torch.acos(cos)
    s_angle = torch.sin(angle)  # sine of angle
    s_angle += (s_angle == 0) * torch.finfo(torch.float32).eps  # for numerical stability

    interpolated = torch.sin((1.0 - c) * angle) / s_angle * tensor1 + torch.sin(c * angle) / s_angle * tensor2

    return interpolated.unsqueeze(0)

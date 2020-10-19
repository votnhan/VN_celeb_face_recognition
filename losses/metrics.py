import torch

def accuracy(output, target):
    cls_output = torch.argmax(output, dim=1)
    match = (cls_output == target).sum().item()
    bz = output.size(0)
    return match / bz

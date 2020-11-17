import torch
from torch import nn


def freeze_conv1d_params(layer, weight_indices, bias_indices=None):

    raise NotImplementedError("this function will work fine once PyTorch make gradients consistent between convnd")

    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, nn.Conv1d):
        raise ValueError("layer must be a valid Conv1d layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError("weight_indices must be less than the number output channels")

    if max(bias_indices) >= layer.bias.shape[0]:
        raise IndexError("bias_indices must be less than the number output channels")

    def freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier):
        return grad_input[0], grad_input[1] * weight_multiplier, grad_input[2]*bias_multiplier

    weight_multiplier = torch.ones(layer.weight.shape[0])
    weight_multiplier[weight_indices] = 0
    weight_multiplier = weight_multiplier.view(-1, 1, 1)
    bias_multiplier = torch.ones(layer.weight.shape[0])
    bias_multiplier[bias_indices] = 0
    freezing_hook = lambda layer, grad_input, grad_output: freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier)

    layer.register_backward_hook(freezing_hook)


def freeze_conv2d_params(layer, weight_indices, bias_indices=None):
    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, nn.Conv2d):
        raise ValueError("layer must be a valid Conv2d layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError("weight_indices must be less than the number output channels")

    if max(bias_indices) >= layer.bias.shape[0]:
        raise IndexError("bias_indices must be less than the number output channels")

    def freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier):
        return grad_input[0], grad_input[1] * weight_multiplier, grad_input[2] * bias_multiplier

    weight_multiplier = torch.ones(layer.weight.shape[0])
    weight_multiplier[weight_indices] = 0
    weight_multiplier = weight_multiplier.view(-1, 1, 1, 1)
    bias_multiplier = torch.ones(layer.weight.shape[0])
    bias_multiplier[bias_indices] = 0
    freezing_hook = lambda layer, grad_input, grad_output: freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier)

    layer.register_backward_hook(freezing_hook)


def freeze_conv3d_params(layer, weight_indices, bias_indices=None):
    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, nn.Conv3d):
        raise ValueError("layer must be a valid Conv3d layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError("weight_indices must be less than the number output channels")

    if max(bias_indices) >= layer.bias.shape[0]:
        raise IndexError("bias_indices must be less than the number output channels")

    def freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier):
        return grad_input[0], grad_input[1] * weight_multiplier, grad_input[2] * bias_multiplier

    weight_multiplier = torch.ones(layer.weight.shape[0])
    weight_multiplier[weight_indices] = 0
    weight_multiplier = weight_multiplier.view(-1, 1, 1, 1, 1)
    bias_multiplier = torch.ones(layer.weight.shape[0])
    bias_multiplier[bias_indices] = 0
    freezing_hook = lambda layer, grad_input, grad_output: freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier)

    layer.register_backward_hook(freezing_hook)


def freeze_linear_params(layer, weight_indices, bias_indices=None):
    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, nn.Linear):
        raise ValueError("layer must be a valid Linear layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError("weight_indices must be less than the number output channels")

    if max(bias_indices) >= layer.bias.shape[0]:
        raise IndexError("bias_indices must be less than the number output channels")

    def freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier):
        return grad_input[0] * bias_multiplier, grad_input[1], grad_input[2] * weight_multiplier

    weight_multiplier = torch.ones(layer.weight.shape[0])
    weight_multiplier[weight_indices] = 0
    weight_multiplier = weight_multiplier.view(1, -1)
    bias_multiplier = torch.ones(layer.weight.shape[0])
    bias_multiplier[bias_indices] = 0
    freezing_hook = lambda layer, grad_input, grad_output: freezing_hook_full(layer, grad_input, grad_output, weight_multiplier, bias_multiplier)

    layer.register_backward_hook(freezing_hook)

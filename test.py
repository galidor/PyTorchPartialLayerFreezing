import torch
from torch import optim
from torch.nn import functional as F

from partial_freezing import freeze_conv2d_params, freeze_conv1d_params, freeze_conv3d_params, freeze_linear_params
from test_models import net_conv1d, net_conv3d, vgg16_bn


def test_conv1d_freezing(net):
    indices = [0, 1]
    freeze_conv1d_params(net.conv1, indices)

    indices = [2, 3, 4]
    freeze_conv1d_params(net.conv1, indices)

    init_weights = net.conv1.weight.clone()
    init_biases = net.conv1.bias.clone()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    random_input1d = torch.randn((1, 5, 32))

    net.train()
    optimizer.zero_grad()
    output = net(random_input1d)
    loss = F.mse_loss(output, random_input1d)
    loss.backward()

    if (net.conv1.bias.grad[indices] == torch.zeros_like(net.conv1.bias.grad[indices])).all() and \
            (net.conv1.weight.grad[indices, :, :] == torch.zeros_like(
                net.conv1.weight.grad[indices, :, :])).all():
        print("Conv1d grads passed")
    else:
        print("Conv1d grads failed")
    optimizer.step()

    if (net.conv1.bias == init_biases)[indices].all() and (net.conv1.weight == init_weights)[indices].all():
        print("Conv1d frozen weights and biases passed")
    else:
        print("Conv1d frozen weights and biases failed")


def test_conv2d_freezing(net):
    indices = [2, 3, 4, 5]
    for module in net.features:
        if isinstance(module, torch.nn.Conv2d):
            freeze_conv2d_params(module, indices)

    init_weights = net.features[3].weight.clone()
    init_biases = net.features[3].bias.clone()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    random_input2d = torch.randn((1, 3, 32, 32))

    net.train()
    optimizer.zero_grad()
    output = net(random_input2d)
    loss = F.cross_entropy(output, torch.tensor([5]))
    loss.backward(create_graph=True)
    if (net.features[3].bias.grad[indices] == torch.zeros_like(net.features[3].bias.grad[indices])).all() and \
       (net.features[3].weight.grad[indices, :, :, :] == torch.zeros_like(net.features[3].weight.grad[indices, :, :, :])).all():
        print("Conv2d grads passed")
    else:
        print("Conv2d grads failed")
    optimizer.step()

    if (net.features[3].bias == init_biases)[indices].all() and (net.features[3].weight == init_weights)[indices].all():
        print("Conv2d frozen weights and biases passed")
    else:
        print("Conv2d frozen weights and biases failed")


def test_conv3d_freezing(net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    indices = [2, 3, 4]
    freeze_conv3d_params(net.conv1, indices)

    init_weights = net.conv1.weight.clone()
    init_biases = net.conv1.bias.clone()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    random_input3d = torch.randn((1, 5, 32, 32, 32)).to(device)

    net.train()
    optimizer.zero_grad()
    output = net(random_input3d)
    loss = F.mse_loss(output, random_input3d)
    loss.backward()
    if (net.conv1.bias.grad[indices] == torch.zeros_like(net.conv1.bias.grad[indices])).all() and \
            (net.conv1.weight.grad[indices, :, :, :, :] == torch.zeros_like(
                net.conv1.weight.grad[indices, :, :, :, :])).all():
        print("Conv3d grads passed")
    else:
        print("Conv3d grads failed")
    optimizer.step()

    if (net.conv1.bias == init_biases)[indices].all() and (net.conv1.weight == init_weights)[indices].all():
        print("Conv3d frozen weights and biases passed")
    else:
        print("Conv3d frozen weights and biases failed")


def test_linear_freezing(net):
    indices = [2, 3, 4, 5]
    freeze_linear_params(net.classifier[4], indices)

    init_weights = net.classifier[4].weight.clone()
    init_biases = net.classifier[4].bias.clone()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    random_input2d = torch.randn((1, 3, 32, 32))

    net.train()
    optimizer.zero_grad()
    output = net(random_input2d)
    loss = F.cross_entropy(output, torch.tensor([5]))
    loss.backward()

    if (net.classifier[4].bias.grad[indices] == torch.zeros_like(net.classifier[4].bias.grad[indices])).all() and \
            (net.classifier[4].weight.grad[indices, :] == torch.zeros_like(
                net.classifier[4].weight.grad[indices, :])).all():
        print("Linear grads passed")
    else:
        print("Linear grads failed")
    optimizer.step()

    if (net.classifier[4].bias == init_biases)[indices].all() and (net.classifier[4].weight == init_weights)[indices].all():
        print("Linear frozen weights and biases passed")
    else:
        print("Linear frozen weights and biases failed")


if __name__ == '__main__':
    model = net_conv1d()
    test_conv1d_freezing(model)
    model = vgg16_bn()
    test_conv2d_freezing(model)
    model = net_conv3d()
    test_conv3d_freezing(model)
    model = vgg16_bn()
    test_linear_freezing(model)

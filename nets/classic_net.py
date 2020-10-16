import torch.nn as nn

from torchvision.models import resnet18, inception_v3, vgg11


def get_resnet18(pretrained):
    net = resnet18(pretrained=pretrained)
    net.fc = nn.Linear(512, 12)
    return net

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
"""
#kaiming initialization
def init_weights(net):
    for m in net.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)


import torch
import torch.nn as nn

# custom weights initialization
# Reference (PyTorch Tutorials)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Loss for discriminator
def disc_loss(real, fake):
    return torch.mean(fake) - torch.mean(real)

# Loss for generator
def gen_loss(fake):
    return -1.*torch.mean(fake)
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
def disc_loss(real, fake, grad_pen, disc_lambda):
    return torch.mean(fake) - torch.mean(real) + disc_lambda*grad_pen

# Loss for generator
def gen_loss(fake):
    return -1.*torch.mean(fake)

# Gradient penalty alt to Grad Cliping
def get_gradient_penalty(disc_net, real_images, fake_images, epsilon):
    mixed_images = real_images * epsilon + fake_images * (1 - epsilon)
    mixed_scores = disc_net(mixed_images)
    
    gradient = torch.autograd.grad(
                                    inputs=mixed_images,
                                    outputs=mixed_scores,
                                    grad_outputs=torch.ones_like(mixed_scores), 
                                    create_graph=True,
                                    retain_graph=True,
                                )[0]
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty

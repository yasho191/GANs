import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as torch_utils
from discriminator import Discriminator
from generator import Generator
from utils import weights_init
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--DATA_DIR", type=str, required=True, help="Training Data Directory")
parser.add_argument("-m", "--MODEL_PATH", type=str, required=True, help="Model Save Path")
parser.add_argument("-a", "--ANIMATION_PATH", type=str, required=True, help="Animation Save Path")
parser.add_argument("-t", "--TRAINING_PLOT_PATH", type=str, required=True, help="Training Plot Path")

args = parser.parse_args()

manualSeed = 123
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


data_dir = args.DATA_DIR
model_save_path = args.MODEL_PATH
animation_save_path = args.ANIMATION_PATH
training_plot_save_path = args.TRAINING_PLOT_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 64
lr = 0.00005
beta1 = 0.5
batch_size = 64
noise_dim = 100
workers = 2
num_epochs = 50
# Number of discriminator updates for 1 generator update
n_critic = 5
# Clip value
clip = 0.01
# Monitor Progress
progress = list()
fixed_noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)

disc_net = Discriminator()
gen_net = Generator()
disc_net.to(device)
gen_net.to(device)
disc_net.apply(weights_init)
gen_net.apply(weights_init)


dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

dataloader = data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)

# Loss for discriminator
def disc_loss(real, fake):
    return torch.mean(fake) - torch.mean(real)

# Loss for generator
def gen_loss(fake):
    return -1.*torch.mean(fake)

disc_optimizer = optim.RMSprop(disc_net.parameters(), lr=lr)
gen_optimizer = optim.RMSprop(gen_net.parameters(), lr=lr)


# Training Loop

# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        real_images = data[0].to(device)
        num_images = real_images.size(0)  
        noise = torch.randn(num_images, noise_dim, 1, 1, device=device)

        # Training the discriminator (n_critic)
        # Train Discriminator on Real Images and Fake Images
        # Backpropagate the loss
        mean_disc_err = 0.
        for _ in range(n_critic):
            disc_net.zero_grad()

            real_output = disc_net(real_images).view(-1)

            fake = gen_net(noise)
            fake_output = disc_net(fake.detach()).view(-1)

            disc_err = disc_loss(real_output, fake_output)
            mean_disc_err += disc_err.item()
            disc_err.backward()
            disc_optimizer.step()

            # Clip weights of discriminator
            for p in disc_net.parameters():
                p.data.clamp_(-clip, clip)

        # Training the Generator - Every n_critic iterations
        # Steps:
        # 1. Get Discriminator Predictions on Fake Images
        # 2. Calculate loss
        gen_net.zero_grad()
        gen_images = gen_net(noise)

        output = disc_net(gen_images).view(-1)
        gen_err = gen_loss(output)

        gen_err.backward()
        gen_optimizer.step()

        # Training Update
        if i % 50 == 0:
            print(
                f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {disc_err.item()}\tLoss_G: {gen_err.item()}"
            )

        # Tracking loss
        G_losses.append(gen_err.item())
        D_losses.append(mean_disc_err/n_critic)

        # Tracking Generator Progress
        if (iters % 500 == 0) or (
            (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
        ):
            with torch.no_grad():
                fake = gen_net(fixed_noise).detach().cpu()
            progress.append(torch_utils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# Save generator
torch.save(gen_net, model_save_path)

# Plot Training Graph
fig1 = plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(training_plot_save_path)
plt.show()

# Progress Animation
fig2 = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in progress]
anim = animation.ArtistAnimation(fig2, ims, interval=1000, repeat_delay=1000, blit=True)
writervideo = animation.FFMpegWriter(fps=60)
anim.save(animation_save_path, writer=writervideo)
plt.close()
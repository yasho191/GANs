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
lr = 0.0002
beta1 = 0.5
batch_size = 64
noise_dim = 100
workers = 2
num_epochs = 5
real_label = 1.0
fake_label = 0.0
classes = 10
# Monitor Progress
progress = list()
fixed_noise = torch.randn(100, noise_dim, device=device)
fixed_labels = torch.Tensor([[i]*10 for i in range(10)]).view(100,).int().to(device)

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


criterion = nn.BCELoss()

disc_optimizer = optim.Adam(disc_net.parameters(), lr=lr, betas=(beta1, 0.999))
gen_optimizer = optim.Adam(gen_net.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        real_images = data[0].to(device)
        real_labels = data[1].to(device)
        num_images = real_images.size(0)
        
        real_target = torch.ones(num_images,).to(device)
        fake_target = torch.zeros(num_images,).to(device)
        
        # Training the discriminator
        # Train Discriminator on Real Images and Fake Images
        disc_net.zero_grad()

        real_output = disc_net(real_images, real_labels).view(-1)
        disc_err_real = criterion(real_output, real_target)
        
        # Conditional Noise
        noise = torch.randn(num_images, noise_dim, device=device)
        noise_labels = torch.randint(0, classes, (num_images, ), device=device)

        fake = gen_net(noise, noise_labels)

        fake_output = disc_net(fake.detach(), noise_labels).view(-1)
        disc_err_fake = criterion(fake_output, fake_target)

        disc_err = (disc_err_real + disc_err_fake)/2
        disc_err.backward()
        disc_optimizer.step()

        # Training the Generator
        # Steps:
        # 1. Get Discriminator Predictions on Fake Images
        # 2. Calculate loss
        gen_net.zero_grad()
        
        output = disc_net(fake, noise_labels).view(-1)

        gen_err = criterion(output, real_target)
        gen_err.backward()
        gen_optimizer.step()

        # Training Update
        if i % 50 == 0:
            print(
                f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {disc_err.item()}\tLoss_G: {gen_err.item()}"
            )

        # Tracking loss
        G_losses.append(gen_err.item())
        D_losses.append(disc_err.item())

        # Tracking Generator Progress
        if (iters % 10 == 0) or (
            (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
        ):
            with torch.no_grad():
                fake = gen_net(fixed_noise, fixed_labels).detach().cpu()
            progress.append(torch_utils.make_grid(fake, padding=2, nrow=10, normalize=True))

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
writervideo = animation.FFMpegWriter(fps=5)
anim.save(animation_save_path, writer=writervideo)
plt.close()
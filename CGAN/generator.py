import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, classes):
        super(Generator, self).__init__()
        self.classes = classes

        self.embedding = nn.Sequential(
            nn.Embedding(classes, 64),
            nn.Unflatten(1, (1, 8, 8))
        )

        self.latent_vector = nn.Sequential(
            nn.Linear(100, 4096),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 8, 8))
        )

        upsample_1 = self.upsample_block(65, 256, 1)
        upsample_2 = self.upsample_block(256, 128, 1)
        upsample_3 = self.upsample_block(128, 64, 1)

        self.conv_model = nn.Sequential(
            upsample_1,
            upsample_2,
            upsample_3,
            nn.Conv2d(64, 3, (1, 1), 1, 0),
            nn.Tanh()
        )
    
    def upsample_block(self, in_channels, out_channels, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (4, 4), 2, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, label):
        latent_vector = self.latent_vector(x)
        label_embedding = self.embedding(label)
        comb_latent_vector = torch.concat((latent_vector, label_embedding), dim = 1)
        output = self.conv_model(comb_latent_vector)
        return output
    
# Test Generator
if __name__ == "__main__":
    x = torch.rand((2, 100))
    label = torch.randint(0, 4, (2,))
    net = Generator(4)
    output = net(x, label)
    print(output.shape)
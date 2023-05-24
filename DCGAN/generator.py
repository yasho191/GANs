import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        upsample_1 = self.upsample_block(100, 1024, 0)
        upsample_2 = self.upsample_block(1024, 512, 1)
        upsample_3 = self.upsample_block(512, 256, 1)
        upsample_4 = self.upsample_block(256, 128, 1)

        self.gen = nn.Sequential(
            upsample_1,
            upsample_2,
            upsample_3,
            upsample_4,
            nn.ConvTranspose2d(128, 3, (4, 4), 2, 1),
            nn.Tanh(),
        )

    def upsample_block(self, in_channels, out_channels, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (4, 4), 2, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.gen(x)


# Test Generator
if __name__ == "__main__":
    x = torch.rand((1, 100, 1, 1))
    net = Generator()
    output = net(x)
    print(output.shape)

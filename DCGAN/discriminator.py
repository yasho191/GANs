import torch
import torch.nn as nn

# Discriminator
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        conv_1 = self.conv_block(3, 64)
        conv_2 = self.conv_block(64, 128)
        conv_3 = self.conv_block(128, 256)
        conv_4 = self.conv_block(256, 512)

        self.classifier = nn.Sequential(
            conv_1,
            conv_2,
            conv_3,
            conv_4,
            nn.Conv2d(512, 1, (5, 5), 2, 1),
            nn.Sigmoid(),
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (5, 5), 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.classifier(x)


# Test Discriminator
if __name__ == "__main__":
    x = torch.rand((1, 3, 64, 64))
    net = Discriminator()
    out = net(x)
    print(out.shape)

import torch
import torch.nn as nn

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, classes) -> None:
        super(Discriminator, self).__init__()
        self.classes = classes

        self.embedding = nn.Sequential(
            nn.Embedding(classes, 64),
            nn.Linear(64, 64*64),
            nn.Unflatten(1, (1, 64, 64))
        )

        conv_1 = self.conv_block(4, 64)
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
    
    def forward(self, x, label):
        label_embedding = self.embedding(label)
        comb_latent_vector = torch.concat((x, label_embedding), dim=1)
        output = self.classifier(comb_latent_vector)
        return output
    
# Test Discriminator
if __name__ == "__main__":
    x = torch.rand((2, 3, 64, 64))
    label = torch.randint(0, 4, (2,))
    net = Discriminator(4)
    out = net(x, label)
    print(out.shape)
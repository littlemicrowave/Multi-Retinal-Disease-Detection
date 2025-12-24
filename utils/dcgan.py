import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, output_channels=3):
        super().__init__()

        self.fc = nn.Linear(z_dim, 8 * 8 * 1024)

        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv_out = nn.ConvTranspose2d(64, output_channels, kernel_size=5, stride=1, padding=2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self._initialize_weights()

    def forward(self, z):
        # z: (batch_size, z_dim)
        x = self.fc(z)
        x = x.view(-1, 1024, 8, 8)
        x = self.activation(x)

        x = self.activation(self.bn1(self.deconv1(x)))
        x = self.activation(self.bn2(self.deconv2(x)))
        x = self.activation(self.bn3(self.deconv3(x)))
        x = self.activation(self.bn4(self.deconv4(x)))

        x = torch.tanh(self.deconv_out(x))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(8 * 8 * 1024, 1)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.initialize_weights()

    def forward(self, x):
        # x: (B, 3, 128, 128)
        x = self.activation(self.bn1(self.conv1(x)))   # 64x64x64
        x = self.activation(self.bn2(self.conv2(x)))   # 32x32x128
        x = self.activation(self.bn3(self.conv3(x)))   # 16x16x256
        x = self.activation(self.bn4(self.conv4(x)))   # 16x16x512
        x = self.activation(self.bn5(self.conv5(x)))   # 8x8x1024

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        out = torch.sigmoid(logits)

        return out, logits

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


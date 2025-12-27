import torch
import torch.nn as nn
import torch.nn.functional as F
from .cvae import FiLM
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim, output_channels=3):
        super().__init__()

        self.fc = nn.Linear(z_dim, 8 * 8 * 512)

        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film1 = FiLM(label_dim, 256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film2 = FiLM(label_dim, 128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film3 = FiLM(label_dim, 64)

        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.film4 = FiLM(label_dim, 64)

        #self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.film5 = FiLM(label_dim, 16)

        self.deconv_out = nn.ConvTranspose2d(64, output_channels, kernel_size=5, stride=1, padding=2)
        self.activation = nn.LeakyReLU(0.2)

        self.initialize_weights()

    def forward(self, z, labels):
        # z: (batch_size, z_dim)
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.activation(x)

        x = self.activation(self.film1(self.deconv1(x), labels))
        x = self.activation(self.film2(self.deconv2(x), labels))
        x = self.activation(self.film3(self.deconv3(x), labels))
        x = self.activation(self.film4(self.deconv4(x), labels))
      #  x = self.activation(self.film5(self.deconv5(x), labels))

        x = torch.tanh(self.deconv_out(x))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_channels = 32, label_dim = 3):
        super().__init__()
    
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3)
       # self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        #self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        #self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2))
        #self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1))
        #self.bn5 = nn.BatchNorm2d(256)

        #self.conv6 = spectral_norm(nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2))
        #self.bn6 = nn.BatchNorm2d(512)

       # self.conv_rf = spectral_norm(nn.Conv2d(1024, 1, kernel_size=1))
      #  self.patch_head = spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1))
        self.fc = spectral_norm(nn.Linear(512, 1))
        self.cond = spectral_norm(nn.Linear(label_dim, 512))
        self.aux = spectral_norm(nn.Linear(512, label_dim))
        self.activation = nn.LeakyReLU(0.2)

        self.initialize_weights()

    def forward(self, x, labels):
        # x: (B, 3, 128, 128)
        x = self.activation(self.conv1(x))   # 64x64x64
        x = self.activation(self.conv2(x))   # 32x32x128
        x = self.activation(self.conv3(x))   # 16x16x256
        x = self.activation(self.conv4(x))   # 16x16x512
        out = self.activation(self.conv5(x))   # 8x8x1024
      #  x = self.activation(self.conv6(x))   # 8x8x1024

        x = torch.sum(out, dim=(2, 3))
       # patch_out = self.patch_head(out)
        logit = self.fc(x)
       # logits = self.conv_rf(x)
        y = self.cond(labels)#[:, :, None, None]
        proj = torch.sum(y * x, dim=1, keepdim=True)
        aux_logits = self.aux(x)
        return logit + proj, aux_logits

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

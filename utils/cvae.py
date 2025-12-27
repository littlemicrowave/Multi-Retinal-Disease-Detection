import torch.nn as nn
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

#defaults
BATCH = 32
IMG_SIZE = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet_dir = "pretrained_backbone/ckpt_resnet18_ep50.pt"
train_images= "images/train"
train_labels = "train.csv"
val_images = "images/val"
val_labels = "val.csv"

transform_in = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_out = transforms.Compose([
    transforms.Resize((IMG_SIZE //2, IMG_SIZE//2)),
    transforms.ToTensor()
])

''''
def inject_condition(x, labels):
    _, d = labels.shape
    b, _, h, w = x.shape
    labels = torch.reshape(labels, [b, d, 1, 1])
    labels = labels.expand([b, d, h, w])
    x = torch.cat([x, labels], dim = 1)
    return x
'''

class VAEDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform_in, transform_out):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform_in = transform_in
        self.transform_out = transform_out

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        img_in = self.transform_in(img)
        img_out = self.transform_out(img)
        return img_in, labels, img_out

class Encoder(nn.Module):
    def __init__(self, backbone_path = None, latent_channels = 32, label_dim = None):
        super().__init__()
        resnet = models.resnet18(weights = None)
        out_channels = resnet.fc.in_features
        resnet.fc = nn.Linear(out_channels, 3)
        if backbone_path != None:
           resnet.load_state_dict(torch.load(backbone_path), strict=False)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.mu = nn.Conv2d(out_channels, latent_channels, 3, 2, 1)  ## B, 32, 4, 4
        self.logvar = nn.Conv2d(out_channels, latent_channels, 3, 2, 1) # B, 32, 4, 4
        if label_dim != None:
            self.film = FiLM(label_dim, out_channels)
    

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def reparameterize(self, mu, log_var):
        # standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        #  random noise using the same shape as std
        eps = torch.randn_like(std)
        # reparameterized sample
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.logvar(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Encoded output, mean, and log variance
        return z, mu, log_var
    
    def forward(self, x, labels):
        encoded = self.encoder(x)
        encoded = self.film(encoded, labels)
        mu = self.mu(encoded)
        log_var = self.logvar(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Encoded output, mean, and log variance
        return z, mu, log_var
    
    

class FiLM(nn.Module):
    '''
    Feature-wise Linear Modulation. A FiLM
    layer carries out a simple, feature-wise affine transformation
    on a neural network's intermediate features, conditioned on
    an arbitrary input. FiLM layers enable input to influence Convolutional Neural Network (CNN)
    computation over an image. #https://arxiv.org/pdf/1709.07871 
    '''
    def __init__(self, label_dim, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        if label_dim != None:
            self.g = nn.Linear(label_dim, channels)
            self.b = nn.Linear(label_dim, channels)
            # BigGAN centering
            nn.init.zeros_(self.g.weight)
            nn.init.ones_(self.g.bias)
            nn.init.zeros_(self.b.weight)
            nn.init.zeros_(self.b.bias)

    def forward(self, x, labels):
        if labels != None:
            gamma = self.g(labels)
            beta = self.b(labels)
            gamma = gamma[:, :, None, None]       # (B, C, 1, 1)
            beta  = beta[:, :, None, None]        # (B, C, 1, 1)
            x = self.bn(x)
            x = gamma * x + beta   
        else:
            x = self.bn(x)
        return x 
    
class ResFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, label_dim = None, upsample = True):
        super().__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.upsample = nn.Identity()
        
        self.film1 = FiLM(label_dim, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.film2 = FiLM(label_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x, labels):
        #skip path
        skip = self.skip(self.upsample(x))

        #main path
        h = self.film1(x, labels)
        h = self.relu(h)
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.film2(h, labels)
        h = self.relu(h)
        h = self.conv2(h)

        return h + skip


class Decoder(nn.Module):
    def __init__(self, latent_channels=32, base_channels=512, label_dim=3, num_upsamples=5): # 8, 16, 32, 64, 128
        super().__init__()


        blocks_inject = []

        in_ch = latent_channels
        out_ch = base_channels

        # first
        blocks_inject.append(ResFiLM(in_ch, out_ch, label_dim))
        in_ch = out_ch
        num_upsamples = num_upsamples - 1

        for _ in range(num_upsamples):
            out_ch = in_ch // 2
            blocks_inject.append(ResFiLM(in_ch, out_ch, label_dim))
            in_ch = out_ch

        self.blocks_inject = nn.ModuleList(blocks_inject)
        self.final_block = ResFiLM(out_ch, out_ch//2, upsample=False)
        self.relu = nn.LeakyReLU()
        self.to_rgb = nn.Conv2d(out_ch//2, 3, 5, padding=2)

    def forward(self, x, labels):
        for block in self.blocks_inject:
            x = block(x, labels)

        x = self.final_block(x, None)
        x = self.relu(x)

        return self.to_rgb(x)

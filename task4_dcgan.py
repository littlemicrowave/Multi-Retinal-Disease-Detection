from utils.train_eval import RetinaMultiLabelDataset, train_dcgan, train_images, train_labels, device, sample, save_samples
from utils.dcgan import Generator, Discriminator
from torch import nn
from torch import optim
from torchvision import transforms
from torch import load
import torch 
BATCH = 16
SIZE = 128
NZ = 128


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
 #   transforms.ColorJitter(brightness=0.3, saturation = 0.1, hue = 0.1),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

G = Generator(z_dim=NZ, label_dim=3).to(device)
D = Discriminator().to(device)

G.load_state_dict(load("task4/generator.pt"))
D.load_state_dict(load("task4/discriminator.pt"), strict=True)

'''
train_true = RetinaMultiLabelDataset(train_labels, train_images, transform=transform)

G_optim = optim.Adam(G.parameters(), lr = 5e-5, betas=(0.5, 0.999))
D_optim = optim.Adam(D.parameters(), lr = 1e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()
train_dcgan(NZ,BATCH, G, D, G_optim, D_optim, train_true, 1000, grid_size=[1], inject_noise=False, hinge=True)#1500
'''
labels = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]],dtype=torch.float).to(device)
samples = sample(4, G, labels, [NZ], temp=1, seed=5, n= 3)
save_samples(samples, labels, "task4/generated", denorm=True)
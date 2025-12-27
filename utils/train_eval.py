import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import os
from torchsummary import summary
from torchvision import models, transforms
from tqdm import tqdm
from utils.image import *

from .blocks import *

# reproducibility
torch.manual_seed(0)

#defaults
BATCH = 32
IMG_SIZE = 256
label_names = ["D", "G", "A"]
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet_dir = "pretrained_backbone/ckpt_resnet18_ep50.pt"
train_images= "images/train"
train_labels = "train.csv"
val_images = "images/val"
val_labels = "val.csv"
offsite_test_images = "images/offsite_test"
offsite_test_labels = "offsite_test.csv"
onsite_test_images = "images/onsite_test"
onsite_test_labels = "onsite_test_submission.csv"


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels


class FeaturesPlusSE(nn.Module):
    def __init__(self, features, ch=1280, ratio=16):
        super().__init__()
        self.features = features
        self.se = SEBlock(ch, ratio)

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        return x

    
class Classifier(nn.Module):
    def __init__(self, backbone = "resnet", block=None ,dir = None):
        super().__init__()

        if dir == None:
            raise FileNotFoundError
        layers = torch.load(dir)

        if backbone == "resnet":
            self.model = models.resnet18()
            self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        elif backbone == "efficientnet":
            self.model = models.efficientnet_b0()
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 3)
        else:
            raise ValueError("Unsupported backbone")
        self.model.load_state_dict(layers)

        if block == "se":
            self.model.features = FeaturesPlusSE(self.model.features)
        elif block == "mha":
            self.model.features.add_module("mha", MultiHeadAttentionCNN(H=8, W=8, channels=1280, num_heads=4, projection_dim=256, use_rpe=True, max_rpe_dist=3))
    
    def forward(self, X):
        return self.model(X)
    

class Resnet_MHA_SE(nn.Module):
    def __init__(self, block = None, backbone_dir = None):
        super().__init__()
        if block not in ["se", "mha"]:
            raise ValueError
        if backbone_dir == None:
            raise FileNotFoundError
        layers = torch.load(backbone_dir)
        self.model = models.resnet18()
        channels = self.model.fc.in_features
        self.model.fc = nn.Linear(channels, 3)
        self.model.load_state_dict(layers)
        if block == "se":
            self.model.layer4.add_module(block, SEBlock(channels, ratio=16))
        if block == "mha":
            self.model.layer4.add_module(block, MultiHeadAttentionCNN(H=8, W=8, channels=channels, num_heads=4, projection_dim=128, use_rpe=True, max_rpe_dist=3))
    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = False
    def unfreeze_module(self, name: str):
        for n, module in self.model.named_modules():
            if name in n:
                for p in module.parameters():
                    p.requires_grad = True 
    def forward(self, X):
        return self.model(X)
        

def eval_model(model, dataset, csv_file = None, report_dir = None, cam=None, cam_max_batches=None, shuffle=False):
    loader = DataLoader(dataset, BATCH, shuffle=shuffle)
    preds = []
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm(loader):
            output = (nn.functional.sigmoid(model(X.to(device))) > 0.5).long()
            preds.extend(output.cpu().numpy())

    preds = np.stack(preds)
    if report_dir:
        cr = classification_report(dataset.data[label_names].to_numpy(), preds, target_names=label_names, zero_division= np.nan)
        print(cr)
        with open(report_dir, "w") as f:
            f.write(cr)
    if csv_file:
        data = dataset.data.copy()
        data[label_names] = preds
        data.to_csv(csv_file, index = False)

    if cam is not None:
      generated = 0
      images, targets, preds, cams = [], [], [], {}
      for X, Y in tqdm(loader):
        X = X.to(device)
        with torch.enable_grad():
          model.zero_grad(set_to_none=True)
          logits = model(X)
          output = (nn.functional.sigmoid(logits) > 0.5).long()

          for class_idx in range(len(label_names)):
            model.zero_grad(set_to_none=True)

            score = logits[:, class_idx].sum()
            score.backward(retain_graph=True)

            cam_map = cam.compute_cam()   # HxW in [0,1]
            if class_idx not in cams:
              cams[class_idx] = []
            cams[class_idx].extend(cam_map)


          preds.extend(output.cpu().numpy())
          images.extend(X.cpu().numpy())
          targets.extend(Y)

        generated += 1
        if cam_max_batches and generated >= cam_max_batches:
          break
      return np.stack(images), cams, np.stack(targets), np.stack(preds)


def train_model(model, train_data, eval_data, optimizer, criterion, epochs, stepLR = None, save_as = None, monitor = "loss"):
    train_loader = DataLoader(train_data,  BATCH, shuffle=True)
    val_loader = DataLoader(eval_data, BATCH, shuffle=False)
    train_size = len(train_data.data)
    eval_size = len(eval_data.data)

    train_losses = []
    val_losses = []

    f1 = []
    accuracy = []
    best_score = np.inf
    if monitor == "f1":
        best_score = -1
    for i in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        val_f1 = 0
        val_accuracy = 0

        for (X, Y) in tqdm(train_loader, desc = "Training"):
            if device == "cuda":
                X = X.to(device)
                Y = Y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        train_loss = train_loss / train_size

        model.eval()
        preds = []
        with torch.no_grad():
            for (X, Y) in tqdm(val_loader, desc="Validation"):
                if device == "cuda":
                    X = X.to(device)
                    Y = Y.to(device)
                output = model(X)
                loss = criterion(output, Y)
                val_loss += loss.item() * X.size(0)
                output = nn.functional.sigmoid(output)
                preds.extend((output > 0.5).cpu().long().numpy())
        
        preds = np.stack(preds)

        val_loss = val_loss / eval_size
        val_accuracy = accuracy_score(eval_data.data[label_names].to_numpy(), preds)
        val_f1 = f1_score(eval_data.data[label_names].to_numpy(), preds, average="macro")

        print(f"Epoch: {i} - Train Loss: {train_loss:2f} - Val Loss: {val_loss:2f} - Val Accuracy: {val_accuracy:2f} - Val F1 (macro): {val_f1:2f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #saving model if score imporved
        improved = False
        if monitor == 'f1' and best_score < val_f1:
            improved = True
            best_score = val_f1
        elif monitor == "loss"  and best_score > val_loss:
            improved = True
            best_score = val_loss
            

        if improved:
            print("Model improved! Saving if save_as is set.")
            if save_as != None:
                torch.save(model.state_dict(), save_as)

        f1.append(val_f1)
        accuracy.append(val_accuracy)
        if stepLR != None:
                stepLR.step()
    if monitor == None:
        print("Model saved.")
        torch.save(model.state_dict(), save_as)
    return {"train_loss": train_losses, "val_loss": val_losses, "f1": f1, "accuracy": accuracy, "epochs": epochs}

def training_graphs(results, save_dir):
    x = range(0, results["epochs"])
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x, results["train_loss"], label = "Train Loss")
    plt.plot(x, results["val_loss"], label = "Val Loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("BCE Loss")
    plt.subplot(1, 3, 2)
    plt.plot(x, results["f1"])
    plt.title("Val F1 (macro)")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.subplot(1, 3, 3)
    plt.plot(x, results["accuracy"])
    plt.title("Val accuracy")
    plt.ylabel("Metric")
    plt.xlabel("Epoch")
    fig.savefig(save_dir)


def train_vae(encoder, decoder, elbo: callable, train_data, eval_data, optimizer, epochs, beta_max=1.0, kl_warmup_epochs=15, print_stats_every =10,  freeze_encoder_for = 10, stepLR=None,save_as=None):
    pos = train_data.data[label_names].sum(axis = 0).to_numpy()
    neg = len(train_data.data) - pos
    class_weights = torch.tensor(neg / pos,  dtype=torch.float32)
    labels = torch.tensor(train_data.data[label_names].to_numpy())
    sample_weights = (class_weights * labels).sum(dim = 1)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_data, BATCH, sampler=sampler)
    val_loader = DataLoader(eval_data, BATCH, shuffle=False)
    train_size = len(train_data)
    val_size = len(eval_data)

    train_losses = []
    recon_losses = []
    kl_losses = []

    val_losses = []
    val_recons = []
    val_kls = []

    mus = []
    vars = []

    best_loss = float("inf")
    encoder.freeze_backbone()
    encoder_frozen = True
    print("Backbone frozen.")
    for epoch in range(epochs):

        if encoder_frozen and epoch >= freeze_encoder_for:
            encoder.unfreeze_backbone()
            encoder_frozen = False
            print("Backbone unfrozen.")
        
        encoder.train()
        decoder.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        beta = min(beta_max, beta_max * epoch / kl_warmup_epochs)

        for (X, labels, Y) in tqdm(train_loader, desc=f"Train {epoch}"):
            X = X.to(device)
            Y = Y.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            z, mu, logvar = encoder(X, labels)
            recon = decoder(z, labels)

            loss, recon_loss, kl_loss = elbo(recon, Y, mu, logvar, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_recon += recon_loss.item() * X.size(0)
            total_kl += kl_loss.item() * X.size(0)

        total_loss /= train_size
        total_recon /= train_size
        total_kl /= train_size

        train_losses.append(total_loss)
        recon_losses.append(total_recon)
        kl_losses.append(total_kl)

        encoder.eval()
        decoder.eval()

        val_loss = val_recon = val_kl = 0.0

        with torch.no_grad():
            for X, labels, Y in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                labels = labels.to(device)

                z, mu, logvar = encoder(X, labels)
                recon = decoder(z, labels)

                loss, recon_loss, kl_loss = elbo(recon, Y, mu, logvar, beta)


                mus.append(mu.mean().item())
                vars.append(logvar.exp().mean().item())
                val_loss  += loss.item() * X.size(0)
                val_recon += recon_loss.item() * X.size(0)
                val_kl    += kl_loss.item() * X.size(0)

        val_loss  /= val_size
        val_recon /= val_size
        val_kl    /= val_size
        val_losses.append(val_loss)
        val_recons.append(val_recon)
        val_kls.append(val_kl)

        if epoch % print_stats_every == 0:
            print( f"[Latent stats] \n E[mu]={sum(mus)/len(mus):.4f} E[var]={sum(vars)/len(vars):.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            if save_as is not None:
                torch.save(encoder.state_dict(), save_as + "_encoder.pt")
                torch.save(decoder.state_dict(), save_as + "_decoder.pt")
                print("Model improved, saved.")

        print(f"Train: {total_loss:.4f} (Recon {total_recon:.4f}, KL {total_kl:.4f}) | beta={beta:.3f} --------- Val: {val_loss:.4f} (Recon {val_recon:.4f}, KL {val_kl:.4f})")

        if stepLR is not None:
            stepLR.step()

    return {"train_loss": train_losses, "recon_loss": recon_losses, "kl_loss": kl_losses, "epochs": epochs, "val_loss": val_losses, "val_kl": val_kls, "val_recon": val_recons}

def training_graphs_vae(results, save_dir):
    x = range(results["epochs"])

    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x, results["train_loss"], label="Train ELBO")
    plt.plot(x, results["val_loss"], label = "Val ELBO")
    plt.legend()
    plt.title("ELBO Loss")
    plt.subplot(1, 3, 2)
    plt.plot(x, results["recon_loss"], label = "Train reconstruction")
    plt.plot(x, results["val_recon"], label = "Val reconstruction")
    plt.legend()
    plt.title("Reconstruction Loss (L1)")
    plt.subplot(1, 3, 3)
    plt.plot(x, results["kl_loss"], label = "Train KLD")
    plt.plot(x, results["val_kl"], label = "Val KLD")
    plt.legend()
    plt.title("KL Divergence")

    fig.savefig(save_dir)

def denorm(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    std  = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return x * std + mean

def visualize_reconstructions(encoder, decoder, dataset, device, num_samples=4, path = "task4/reconstruction", denorm_input = False):
    encoder.eval()
    decoder.eval()
    idxs = np.random.choice(len(dataset), size=num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            x_in, labels, x_target = dataset[idx]

            x_in = x_in.unsqueeze(0).to(device)         # (1,3,256,256)
            x_target = x_target.unsqueeze(0).to(device) # (1,3,128,128)
            labels = labels.unsqueeze(0).to(device)

            z, _, _ = encoder(x_in, labels)
            recon = decoder(z, labels)
            recon = torch.sigmoid(recon)

            # move to cpu
            x_target = x_target[0].cpu()
            recon = recon[0].cpu()
            x_in = x_in[0].cpu()
            diff = torch.abs(recon - x_target)
            if denorm_input:
                x_in = denorm(x_in)
            def show(ax, img, title):
                img = img.permute(1, 2, 0).numpy()
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(title)
                ax.axis("off")

            show(axes[row, 0], x_in, f"256x256 {labels} ")
            show(axes[row, 1], x_target, "Target (128x128)")
            show(axes[row, 2], recon, "Reconstruction")
            show(axes[row, 3], diff, "Abs Diff")

    fig.savefig(path)


def save_samples(samples, labels, path = "task4/generation", denorm = False):
    k = len(samples)
    for i in range(k):
        b, _, _, _ = samples[i].shape
        fig, axes = plt.subplots(1, b, figsize=(16, 6))
        labels = labels.cpu()
        for ind in range(b):
            x_out = samples[i][ind].cpu()
            if denorm:
                x_out = (x_out + 1) / 2
            else:
                x_out = torch.sigmoid(x_out)
            def show(ax, img, title):
                img = img.permute(1, 2, 0).numpy()
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(title)
                ax.axis("off")
            show(axes[ind], x_out, f"Generated {labels[ind]}")
        fig.savefig(path + f"_{i}")


def sample(num_samples, decoder, labels, latent_shape = [32,4,4], temp = 0.3, seed = 0, n = 1):
    shape = [num_samples] + latent_shape
    samples = []
    torch.manual_seed(seed)
    while n:
        z = temp*torch.randn(latent_shape).to(device)
        z = z.expand(shape)
        decoder.eval()
        with torch.no_grad():
        # z = temp*torch.randn(shape).to(device)
            samples.append(decoder(z, labels))
        n = n-1
    return samples


def train_dcgan(nz, batch_size, g_network, d_network, g_optimizer, d_optimizer, real_train, epochs, grid_size, inject_noise = True, hinge = False, checkpoint_dir = "task4/", print_images = "task4/generated"):

    pos = real_train.data[label_names].sum(axis = 0).to_numpy()
    neg = len(real_train.data) - pos
    class_weights = torch.tensor(neg / pos,  dtype=torch.float32)
    labels = torch.tensor(real_train.data[label_names].to_numpy())
    sample_weights = (class_weights * labels).sum(dim = 1)


    dataset_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    label_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=batch_size, replacement=True)

    def sample_labels():
        idx = list(label_sampler)
        return labels[idx].to(device)

    train_loader = DataLoader(real_train, batch_size, shuffle=False, sampler=dataset_sampler)
    train_size = len(real_train)
    num_batches = len(train_loader)

    d_losses = []
    g_losses = []
    loss = nn.BCEWithLogitsLoss()
    aux_loss = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        D_loss = 0
        G_loss = 0
        g_network.train()
        d_network.train()
        for (X, Y) in tqdm(train_loader, desc = "Training"):
            d_optimizer.zero_grad()
            X = X.to(device)
            b_size = X.shape[0]
            Y = Y.to(device)
            # maximize log(D(x)) + log(1 - D(G(z)))
            #pass true batch through D
            if inject_noise:
                X = torch.clamp(X + torch.empty_like(X, device=device).normal_(0.0, 0.05), -1, 1)
            if len(grid_size) == 2:
                smooth_true =  torch.empty((b_size,1, grid_size[0], grid_size[1]), device=device).uniform_(0.9, 1.0)
                fake_labels = torch.zeros((b_size,1, grid_size[0], grid_size[1]), dtype=torch.float, device = device)
                true_labels = torch.ones((b_size,1, grid_size[0], grid_size[1]), dtype=torch.float, device = device)
            else:
                smooth_true =  torch.empty((b_size,1), device=device).uniform_(0.9, 1.0)
                fake_labels = torch.zeros((b_size,1), dtype=torch.float, device = device)
                true_labels = torch.ones((b_size,1), dtype=torch.float, device = device)
            d_out, preds= d_network(X, Y)

            if not hinge:
                d_loss_real = loss(d_out, smooth_true)
            else:
                d_loss_real  = F.relu(1 - d_out).mean()
            aux_loss_real = aux_loss(preds, Y)

            #fake batch
            noise = torch.randn((b_size, nz), dtype = torch.float, device = device)
            fake_Y = sample_labels().float()
            g_out = g_network(noise, fake_Y)
            d_out, _ = d_network(g_out.detach(), fake_Y)

            if not hinge:
                d_loss_fake = loss(d_out, fake_labels)
                err_D = 0.5 * (d_loss_fake + d_loss_real)
            else:
                d_loss_fake = F.relu(1 + d_out).mean()
                err_D = (d_loss_fake + d_loss_real) 
           # aux_loss_fake = aux_loss(preds, fake_Y)
            err_D += 0.5 * aux_loss_real #+ 0.25 * aux_loss_fake#+ 0.1*(patch_out_fake + patch_loss_real)


            err_D.backward()
            d_optimizer.step()

            #train G Adversarial
            noise = torch.randn(b_size, nz, device=device)
            fake_Y = sample_labels().float()
            g_out = g_network(noise, fake_Y)
            g_optimizer.zero_grad()
            d_out, aux_fake = d_network(g_out, fake_Y)

            if not hinge: 
                err_G = loss(d_out, true_labels)
            else:
                err_G= (-d_out).mean()
            err_G+= 0.5*aux_loss(aux_fake, fake_Y)# + patch_out_fake*0.1
            
            err_G.backward()
            g_optimizer.step()

            D_loss += err_D.item()
            G_loss += err_G.item()
        
        D_loss /= num_batches
        G_loss /= num_batches
        d_losses.append(D_loss)
        g_losses.append(G_loss)
        print(f"Epoch {epoch} Discriminator loss: {D_loss}, Generator loss: {G_loss}")
        if epoch % 2 == 0 and print_images:
            g_network.eval()
            with torch.no_grad():
                label = real_train[np.random.randint(0, train_size -1)][1].to(device)
                out = g_network(torch.randn((1, nz)).to(device), label.unsqueeze(0))
                out = (out + 1) / 2
                out = out.permute([0, 2, 3, 1])[0].cpu()
                fig = plt.figure(figsize=(8,8))
                plt.imshow(out)
                plt.title(f"{label.cpu().numpy()}")
                plt.axis("off")
                fig.savefig(print_images + f"/image_{epoch}")
        if epoch % 20 == 0 and checkpoint_dir:
            print("Saving.")
            torch.save(g_network.state_dict(), checkpoint_dir + "generator.pt")
            torch.save(d_network.state_dict(), checkpoint_dir + "discriminator.pt")


    return {"epochs": epochs, "g_losses": g_losses, "d_losses": d_losses}


train = RetinaMultiLabelDataset(train_labels, train_images, transform = transform)
val = RetinaMultiLabelDataset(val_labels, val_images, transform = transform)

offsite_test = RetinaMultiLabelDataset(offsite_test_labels, offsite_test_images, transform = transform)
onsite_test = RetinaMultiLabelDataset(onsite_test_labels, onsite_test_images, transform = transform)
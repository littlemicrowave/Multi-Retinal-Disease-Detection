from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import cohen_kappa_score, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import os
from torchsummary import summary
import numpy as np
from .blocks import *

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
    
    def forward(self, X):
        return self.model(X)
        

def eval_model(model, dataset, csv_file = None, report_dir = None):
    loader = DataLoader(dataset, BATCH, shuffle=False)
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


train = RetinaMultiLabelDataset(train_labels, train_images, transform = transform)
val = RetinaMultiLabelDataset(val_labels, val_images, transform = transform)

offsite_test = RetinaMultiLabelDataset(offsite_test_labels, offsite_test_images, transform = transform)
onsite_test = RetinaMultiLabelDataset(onsite_test_labels, onsite_test_images, transform = transform)
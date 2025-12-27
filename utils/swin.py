from torchvision.models import swin_v2_t
from timm.models import tiny_vit
from torch import nn
from .train_eval import *

def add_noise(X):
    return torch.clamp(X + torch.empty_like(X).normal_(0.0, 0.05), -1, 1)
transforms.RandomCrop(15)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=(-90,90)),
    transforms.ColorJitter( brightness=0.35, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    add_noise, 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_swin_data = RetinaMultiLabelDataset(train_labels, train_images, transform=train_transform)

class SwinClassifier(nn.Module):
    def __init__(self, num_classes = 3,  pretrained = True):
        super().__init__()
        if pretrained:
            self.backbone = swin_v2_t(weights ='IMAGENET1K_V1')
        else:
            self.backbone = swin_v2_t()
        self.backbone.head = nn.Linear(768, num_classes)
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
            
    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, X):
        return self.backbone(X)
    
class TinyViT(nn.Module):
    def __init__(self, num_classes = 3,  pretrained = True):
        super().__init__()
        self.backbone = tiny_vit.tiny_vit_11m_224(pretrained=pretrained)
        print(self.backbone)
        self.backbone.head.fc = nn.Linear(448, num_classes)
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
            
    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, X):
        return self.backbone(X)
    
def get_WeightedRandomSampler(data): 
    pos = data.data[label_names].sum(axis = 0).to_numpy()
    neg = len(data.data) - pos
    class_weights = torch.tensor(neg / pos,  dtype=torch.float32)
    labels = torch.tensor(data.data[label_names].to_numpy())
    sample_weights = (class_weights * labels).sum(dim = 1)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True) 

def train_swin(swin, train_data, eval_data, optimizer, criterion, epochs, stepLR = None, save_as = None, monitor = "loss", balanced_sampling = True):
    
    if balanced_sampling:
        train_loader = DataLoader(train_data,  BATCH, sampler=get_WeightedRandomSampler(train_data))
    else:
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
        swin.train()
        train_loss = 0
        val_loss = 0
        val_f1 = 0
        val_accuracy = 0

        for (X, Y) in tqdm(train_loader, desc = "Training"):
            if device == "cuda":
                X = X.to(device)
                Y = Y.to(device)
            optimizer.zero_grad()
            output = swin(X)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        train_loss = train_loss / train_size

        swin.eval()
        preds = []
        with torch.no_grad():
            for (X, Y) in tqdm(val_loader, desc="Validation"):
                if device == "cuda":
                    X = X.to(device)
                    Y = Y.to(device)
                output = swin(X)
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
                torch.save(swin.state_dict(), save_as)

        f1.append(val_f1)
        accuracy.append(val_accuracy)
        if stepLR != None:
                stepLR.step()
    if monitor == None:
        print("Model saved.")
        torch.save(swin.state_dict(), save_as)
    return {"train_loss": train_losses, "val_loss": val_losses, "f1": f1, "accuracy": accuracy, "epochs": epochs}
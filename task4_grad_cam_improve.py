from utils.image import *
from utils.train_eval import *

checkpoints_dir = "trained_models/"
efficient_net_dir = "pretrained_backbone/ckpt_efficientnet_ep50.pt"
model = Classifier(backbone="efficientnet", dir=efficient_net_dir).to(device)

# --- Data Augmentation  ---
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        RandomizeOutsideCircle(p=1.0, radius_ratio=0.48, mode="noise"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train = RetinaMultiLabelDataset(train_labels, train_images, transform = transform)
val = RetinaMultiLabelDataset(val_labels, val_images, transform = transform)

offsite_test = RetinaMultiLabelDataset(offsite_test_labels, offsite_test_images, transform = transform)
onsite_test = RetinaMultiLabelDataset(onsite_test_labels, onsite_test_images, transform = transform)



##Frozen backbone finetuning
params = model.parameters()
for layer in params:
    layer.requires_grad = False
for param in model.model.classifier[1].parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    model.model.classifier[1].parameters(), lr=1e-3, weight_decay=1e-4
)  # 1e-4
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
result = train_model(
    model,
    train,
    val,
    optimizer,
    criterion,
    epochs=20,
    stepLR=scheduler,
    save_as=checkpoints_dir + "efficient_augment_tuned_classifer.pt",
)
training_graphs(result, "task4/efficient_augment")

## Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "efficient_augment_tuned_classifer.pt"))

eval_model(
    model, offsite_test, report_dir="task4/efficient_augment_report_classifier_tuning.txt"
)
## On-site test
eval_model(
    model, onsite_test, csv_file="task4/efficient_augment_submission_classifier_tune.csv"
)

##Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0)  # 5e-4
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.2)

result = train_model(
    model,
    train,
    val,
    optimizer=optimizer,
    criterion=criterion,
    epochs=5,
    stepLR=scheduler,
    save_as=checkpoints_dir + "task4_efficient_augment.pt",
)
training_graphs(result, "task4/efficient_augment_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task4_efficient_augment.pt"))
eval_model(model, offsite_test, report_dir="task4/efficient_augment_report_full_tuning.txt")

##On-siste test
eval_model(model, onsite_test, "task4/efficient_augment_submission_full.csv")

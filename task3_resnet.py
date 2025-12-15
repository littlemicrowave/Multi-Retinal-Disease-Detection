from utils.train_eval import *
from utils.losses import *

resnet_dir = "pretrained_backbone/ckpt_resnet18_ep50.pt"
model = Resnet_MHA_SE(block="mha", backbone_dir= resnet_dir).to(device)
checkpoints_dir = "trained_models/"

print("Resnet + MHA + BCE")
##Stage 1 classifier + mha finetuning
model.freeze_model()
model.unfreeze_module("mha")
for param in model.model.fc.parameters():
        param.requires_grad = True
print(summary(model, (3,IMG_SIZE, IMG_SIZE)))

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay=1e-4)

#Class weighted BCE
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8, gamma=0.8)
result = train_model(model, train, val, optimizer, criterion, epochs=20, stepLR=scheduler, save_as=checkpoints_dir+"resnet_tuned_classifer.pt")
training_graphs(result, "task3/resnet_mha_tuning")

## Off-site test 
model.load_state_dict(torch.load(checkpoints_dir + "resnet_tuned_classifer.pt"))
eval_model(model, offsite_test)#, report_dir="task2/resnet_wbce_report_classifier_tuning.txt")

##Stage 2  Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True
print(summary(model, (3,IMG_SIZE, IMG_SIZE)))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) #5e-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.3) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5) #0.5
result = train_model(model, train, val, optimizer=optimizer, criterion=criterion, epochs=5, stepLR = scheduler, save_as=checkpoints_dir+"task3_mha_resnet.pt", monitor="f1")
training_graphs(result, "task3/resnet_mha_full_tuning")

##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task3_mha_resnet.pt"))
eval_model(model, offsite_test, report_dir = "task3/resnet_mha_report_full_tuning.txt")

##On-siste test export
eval_model(model, onsite_test, "task3/resnet_mha_submission_full.csv")

############################# Same for Focal Loss ##########################

print("Resnet + SE + BCE")
#load clean model
model = Resnet_MHA_SE(block="se", backbone_dir= resnet_dir).to(device)
checkpoints_dir = "trained_models/"


##Stage 1 classifier + se tuning
model.freeze_model()
model.unfreeze_module("se")
for param in model.model.fc.parameters():
        param.requires_grad = True
print(summary(model, (3,IMG_SIZE, IMG_SIZE)))

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8, gamma=0.8)
result = train_model(model, train, val, optimizer, criterion, epochs=20, stepLR=scheduler, save_as=checkpoints_dir+"resnet_tuned_classifer.pt")
training_graphs(result, "task3/resnet_se_tuning")

## Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "resnet_tuned_classifer.pt"))
eval_model(model, offsite_test)

## Stage 2 Full tuning
for layer in model.parameters():
    layer.requires_grad = True
print(summary(model, (3,IMG_SIZE, IMG_SIZE)))

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) #1e-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5) 

result = train_model(model, train, val, optimizer=optimizer, criterion=criterion, epochs=5, stepLR = scheduler, save_as=checkpoints_dir+"task3_se_resnet.pt", monitor="loss")
training_graphs(result, "task3/resnet_se_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task3_se_resnet.pt"))
eval_model(model, offsite_test, report_dir = "task3/resnet_se_report_full_tuning.txt")

##On-siste test export
eval_model(model, onsite_test, "task3/resnet_se_submission_full.csv")

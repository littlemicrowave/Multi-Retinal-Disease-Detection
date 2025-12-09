from utils.train_eval import *
from utils.losses import *

resnet_dir = "pretrained_backbone/ckpt_resnet18_ep50.pt"
model = Classifier(backbone="resnet", dir = resnet_dir).to(device)
checkpoints_dir = "trained_models/"

print("Resnet + weighted BCE")
##Stage 1 backbone finetuning
params = model.parameters()
for layer in params:
        layer.requires_grad = False
for param in model.model.fc.parameters():
        param.requires_grad = True

optimizer = torch.optim.AdamW(model.model.fc.parameters(), lr = 1e-3, weight_decay=1e-4)

#Class weighted BCE
criterion = WeightedBCE(train).to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8, gamma=0.8)
result = train_model(model, train, val, optimizer, criterion, epochs=20, stepLR=scheduler, save_as=checkpoints_dir+"resnet_tuned_classifer.pt")
training_graphs(result, "task2/resnet_wbce_classsifer_tuning")

## Off-site test 
model.load_state_dict(torch.load(checkpoints_dir + "resnet_tuned_classifer.pt"))
eval_model(model, offsite_test)#, report_dir="task2/resnet_wbce_report_classifier_tuning.txt")

##Stage 2  Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True
    
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) #1e-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5) #
result = train_model(model, train, val, optimizer=optimizer, criterion=criterion, epochs=5, stepLR = scheduler, save_as=checkpoints_dir+"task2_wbce_resnet.pt")
training_graphs(result, "task2/resnet_wbce_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task2_wbce_resnet.pt"))
eval_model(model, offsite_test, report_dir = "task2/resnet_wbce_report_full_tuning.txt")

##On-siste test export
eval_model(model, onsite_test, "task2/resnet_wbce_submission_full.csv")


############################# Same for Focal Loss ##########################
print("Resnet + Focal BCE")
#load clean model
model = Classifier(backbone="resnet", dir = resnet_dir).to(device)
checkpoints_dir = "trained_models/"

##Stage 1 backbone finetuning
params = model.parameters()
for layer in params:
        layer.requires_grad = False
for param in model.model.fc.parameters():
        param.requires_grad = True
optimizer = torch.optim.AdamW(model.model.fc.parameters(), lr = 1e-4, weight_decay=1e-4)
#Focal BCE
criterion = FocalLoss(gamma=2, alpha=0.65, reduction="mean", task_type="multi-label")
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8, gamma=0.8)
result = train_model(model, train, val, optimizer, criterion, epochs=20, stepLR=scheduler, save_as=checkpoints_dir+"resnet_tuned_classifer.pt")
training_graphs(result, "task2/resnet_focal_classifier_tuning")

## Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "resnet_tuned_classifer.pt"))
eval_model(model, offsite_test)

for layer in model.parameters():
    layer.requires_grad = True

criterion =  FocalLoss(gamma=2, alpha=0.65, reduction="mean", task_type="multi-label")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) #1e-4

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6) #
result = train_model(model, train, val, optimizer=optimizer, criterion=criterion, epochs=5, stepLR = scheduler, save_as=checkpoints_dir+"task2_focal_resnet.pt", monitor="f1")
training_graphs(result, "task2/resnet_focal_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task2_focal_resnet.pt"))
eval_model(model, offsite_test, report_dir = "task2/resnet_focal_report_full_tuning.txt")

##On-siste test export
eval_model(model, onsite_test, "task2/resnet_focal_submission_full.csv")
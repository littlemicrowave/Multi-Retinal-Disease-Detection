from utils.train_eval import *

resnet_dir = "pretrained_backbone/ckpt_resnet18_ep50.pt"
model = Classifier(backbone="resnet", dir = resnet_dir).to(device)
checkpoints_dir = "trained_models/"
summary(model, (3, 256, 256), device="cuda")

## No fine-tuning: Evaluation directly on ODIR test set

## Off-site test
eval_model(model, offsite_test, report_dir="task1/resnet_report_notuning.txt")

## On-site test
eval_model(model, onsite_test, csv_file="task1/resnet_submission_notune.csv")

##Frozen backbone finetuning
params = model.parameters()
for layer in params:
        layer.requires_grad = False
for param in model.model.fc.parameters():
        param.requires_grad = True
summary(model, (3, 256, 256), device="cuda")

optimizer = torch.optim.AdamW(model.model.fc.parameters(), lr = 1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8, gamma=0.8)
result = train_model(model, train, val, optimizer, criterion, epochs=20, stepLR=scheduler, save_as=checkpoints_dir+"resnet_tuned_classifer.pt")
training_graphs(result, "task1/resnet_classsifer_tuning")

## Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "resnet_tuned_classifer.pt"))
eval_model(model, offsite_test, report_dir="task1/resnet_report_classifier_tuning.txt")
## On-site test
eval_model(model, onsite_test, csv_file="task1/resnet_submission_classifier_tune.csv")

##Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True
    
summary(model, (3, 256, 256), device="cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) #1e-4
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.2) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5) #
result = train_model(model, train, val, optimizer=optimizer, criterion=criterion, epochs=5, stepLR = scheduler, save_as=checkpoints_dir+"task1_resnet.pt")
training_graphs(result, "task1/resnet_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task1_resnet.pt"))
eval_model(model, offsite_test, report_dir = "task1/resnet_report_full_tuning.txt")

##On-siste test
eval_model(model, onsite_test, "task1/resnet_submission_full.csv")
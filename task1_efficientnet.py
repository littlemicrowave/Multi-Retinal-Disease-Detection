from utils.train_eval import *

checkpoints_dir = "trained_models/"
efficient_net_dir = "pretrained_backbone/ckpt_efficientnet_ep50.pt"
model = Classifier(backbone="efficientnet", dir = efficient_net_dir).to(device)
summary(model, (3, 256, 256), device="cuda")

## No fine-tuning: Evaluation directly on ODIR test set

## Off-site test
eval_model(model, offsite_test, report_dir="task1/efficient_report_notuning.txt")

## On-site test
eval_model(model, onsite_test, csv_file="task1/efficient_submission_notune.csv")

##Frozen backbone finetuning
params = model.parameters()
for layer in params:
        layer.requires_grad = False
for param in model.model.classifier[1].parameters():
        param.requires_grad = True
summary(model, (3, 256, 256))
optimizer = torch.optim.AdamW(model.model.classifier[1].parameters(), lr = 1e-3, weight_decay=1e-4) #1e-4
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.8)
result = train_model(model, train, val, optimizer, criterion, epochs=20, stepLR=scheduler, save_as=checkpoints_dir+"efficient_tuned_classifer.pt")
training_graphs(result, "task1/efficient_classifier_tuning")

## Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "efficient_tuned_classifer.pt"))

eval_model(model, offsite_test, report_dir="task1/efficient_report_classifier_tuning.txt")
## On-site test
eval_model(model, onsite_test, csv_file="task1/efficient_submission_classifier_tune.csv")

##Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True
    
summary(model, (3, 256, 256))

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay = 0) #5e-4
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.2)

result = train_model(model, train, val, optimizer=optimizer, criterion=criterion, epochs=5, stepLR = scheduler, save_as=checkpoints_dir+"task1_efficient.pt")
training_graphs(result, "task1/efficient_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task1_efficient.pt"))
eval_model(model, offsite_test, report_dir = "task1/efficient_report_full_tuning.txt")

##On-siste test
eval_model(model, onsite_test, "task1/efficient_submission_full.csv")
from utils.train_eval import *
from utils.losses import *

efficientnet_dir = "pretrained_backbone/ckpt_efficientnet_ep50.pt"
model = Classifier(backbone="efficientnet", block="se", dir=efficientnet_dir).to(device)
checkpoints_dir = "trained_models/"

print("Efficientnet + SE")
 ##Stage 1 backbone + SE finetuning
params = model.parameters()
for layer in params:
    layer.requires_grad = False
for param in model.model.features.se.parameters():
    param.requires_grad = True
for param in model.model.classifier[1].parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
)

criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
result = train_model(
     model,
     train,
     val,
     optimizer,
     criterion,
     epochs=20,
     stepLR=scheduler,
     save_as=checkpoints_dir + "efficientnet_se_tuned_classifer.pt",
 )
training_graphs(result, "task3/efficientnet_se_classsifer_tuning")

## Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "efficientnet_se_tuned_classifer.pt"))
eval_model(model, offsite_test)  # , report_dir="task3/efficientnet_wbce_report_classifier_tuning.txt")

##Stage 2  Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True

optimizer = torch.optim.AdamW(
     [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4
)  # 1e-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer, gamma=0.5
)  # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5) #
result = train_model(
    model,
    train,
    val,
    optimizer=optimizer,
    criterion=criterion,
    epochs=5,
    stepLR=scheduler,
    save_as=checkpoints_dir + "task3_se_efficientnet.pt" )
training_graphs(result, "task3/efficientnet_se_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task3_se_efficientnet.pt"))
eval_model(model, offsite_test, report_dir="task3/efficientnet_se_report_full_tuning.txt")

# ##On-siste test export
eval_model(model, onsite_test, "task3/efficientnet_se_submission_full.csv")

######## MHA #########

model = Classifier(backbone="efficientnet", block="mha", dir=efficientnet_dir).to(
    device
)
checkpoints_dir = "trained_models/"
print("Efficientnet + MHA")
# Stage 1 backbone + MHA finetuning
params = model.parameters()
for layer in params:
    layer.requires_grad = False
for param in model.model.features.mha.parameters():
    param.requires_grad = True
for param in model.model.classifier[1].parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-5
)

criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
result = train_model(
    model,
    train,
    val,
    optimizer,
    criterion,
    epochs=20,
    stepLR=scheduler,
    save_as=checkpoints_dir + "efficientnet_mha_tuned_classifer.pt",
)
training_graphs(result, "task3/efficientnet_mha_classsifer_tuning")

## Off-site test
model.load_state_dict(
    torch.load(checkpoints_dir + "efficientnet_mha_tuned_classifer.pt")
)
eval_model(model, offsite_test)

eval_model(model, onsite_test, "task3/efficientnet_mha_submission_classifier.csv")

##Stage 2  Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
)  # 1e-4
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
criterion = nn.BCEWithLogitsLoss()
result = train_model(
    model,
    train,
    val,
    optimizer=optimizer,
    criterion=criterion,
    epochs=10,
    stepLR=scheduler,
    save_as=checkpoints_dir + "task3_mha_efficientnet.pt", monitor=None #f1 or None 81.5 - 82.23
)
training_graphs(result, "task3/efficientnet_mha_full_tuning")
##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "task3_mha_efficientnet.pt"))
eval_model(
    model, offsite_test, report_dir="task3/efficientnet_mha_report_full_tuning.txt"
)

##On-siste test export
eval_model(model, onsite_test, "task3/efficientnet_mha_submission_full.csv")

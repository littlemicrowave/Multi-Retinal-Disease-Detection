from utils.swin import *
from utils.train_eval import *
#from utils.losses import WeightedBCE, FocalLoss

checkpoints_dir = "task4/"

model = SwinClassifier(num_classes=3, pretrained=True).to(device)
'''
print("SWIN")
##Stage 1 classifier + mha finetuning
model.freeze_backbone()
for param in model.backbone.head.parameters():
        param.requires_grad = True
print(summary(model, (3,IMG_SIZE, IMG_SIZE)))

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, weight_decay=1e-3)

#Class weighted BCE
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma=0.8)
result = train_swin(model, train_swin_data, val, optimizer, criterion, epochs=10, stepLR=scheduler, save_as=checkpoints_dir+"swin_classifer.pt", balanced_sampling=True)
training_graphs(result, "task4/swin_head_tuning")
'''
## Off-site test 
model.load_state_dict(torch.load(checkpoints_dir + "swin_classifer.pt"))
eval_model(model, offsite_test)#, report_dir="task2/resnet_wbce_report_classifier_tuning.txt")

##Stage 2  Full Fine tuning

for layer in model.parameters():
    layer.requires_grad = True
print(summary(model, (3,IMG_SIZE, IMG_SIZE)))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4) #3e-4
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=5, gamma=0.5)
result = train_swin(model, train, val, optimizer=optimizer, criterion=criterion, epochs=30, stepLR = scheduler, save_as=checkpoints_dir+"swin.pt", monitor="f1", balanced_sampling=True) #Either None of F1
training_graphs(result, "task4/swin_full_tuning")

##Off-site test
model.load_state_dict(torch.load(checkpoints_dir + "swin.pt"))
eval_model(model, offsite_test, report_dir = "task4/swin_report_full_tuning.txt")

##On-siste test export
eval_model(model, onsite_test, "task4/swin_submission_full.csv")
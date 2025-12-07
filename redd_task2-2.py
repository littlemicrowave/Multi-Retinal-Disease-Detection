import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):
    def __init__(self,
                 train_dataset,
                 beta=0.999,
                 reduction='mean',
                 task_type='multi-label'):
        """
        Class-Balanced Loss using effective number of samples.

        Args:
            train_dataset: a dataset whose __getitem__ returns (image, label)
                           label must be a tensor of shape (C,) with 0/1 entries.
            beta: float in [0,1). Usually 0.9–0.999.
            reduction: 'none' | 'mean' | 'sum'
            task_type: 'binary', 'multi-class', or 'multi-label'
        """
        super(ClassBalancedLoss, self).__init__()

        assert train_dataset is not None, "You must provide the training dataset."
        assert beta >= 0 and beta < 1, "beta must be in [0, 1)"

        self.beta = beta
        self.reduction = reduction
        self.task_type = task_type

        # obtain lables
        all_labels = []

        for i in range(len(train_dataset)):
            item = train_dataset[i]
            label = item[1]  # assume dataset returns (image, label)
            all_labels.append(label)

        labels_tensor = torch.stack(all_labels, dim=0).float()  # shape (N, C)

        # loss calculations
        samples_per_class = torch.sum(labels_tensor, dim=0).float()
        samples_per_class = torch.clamp(samples_per_class, min=1.0)

        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        class_weights = (1.0 - beta) / effective_num  # shape (C,)
        class_weights = class_weights / class_weights.sum() * len(samples_per_class)

        self.register_buffer("class_weights", class_weights)
        self.num_classes = samples_per_class.shape[0]

    def forward(self, inputs, targets):
        """
        inputs: logits (B, C)
        targets: 0/1 labels (B, C)
        """
        if self.task_type == 'multi-label':
            return self.multi_label_cb_loss(inputs, targets, self.class_weights)
        else:
            raise ValueError("For ODIR you should use task_type='multi-label'.")

    def multi_label_cb_loss(self, inputs, targets, class_weights):
        """
        Multi-label CB loss: class-weighted BCE per label.
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # (B, C)

        w = class_weights.view(1, -1)

        loss = bce * w

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    num_classes = 3
    inputs = torch.randn(16, num_classes)  # Logits from the model
    targets = torch.randint(0, 2, (16, num_classes)).float()  # Ground truth labels

    criterion = ClassBalancedLoss(labels=targets)
    loss = criterion(inputs, targets)

    print(f'Multi-label Class-Balance Loss: {loss.item()}')

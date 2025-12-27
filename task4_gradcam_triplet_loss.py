from utils.image import *
from utils.losses import *
from utils.train_eval import *


def train_model(
    model,
    train_data,
    eval_data,
    optimizer,
    criterion,
    epochs,
    stepLR=None,
    save_as=None,
    monitor="loss",
    *,
    device="cuda",
    label_names=None,
    batch_size=32,
    # --- eXBL / Grad-CAM extras (all optional) ---
    cammer=None,  # object with compute_cam() that returns torch [B,H,W] on device
    img_size=None,  # int, required if cammer is used and compute_cam returns H=W=img_size
    alpha=0.0,  # weight for explanation loss
    expl_triplet_loss=None,  # callable: expl_triplet_loss(X, x_cam, C_good, C_bad) -> scalar tensor
    C_good=None,
    C_bad=None,
    cam_use="gt_pos",  # "all", "gt_pos", "pred_pos", "topk"
    topk=1,
    cam_threshold=0.5,
):
    """
    Multi-label training loop with optional eXBL-style explanation loss.

    cam_use:
      - "all": compute CAM for every label (slow, like your snippet)
      - "gt_pos": CAMs only for labels where Y==1 (cheaper, usually better for multi-label)
      - "pred_pos": CAMs only for predicted positives (sigmoid(logits)>thr)
      - "topk": CAMs only for top-k labels per sample (stable early training)
    """

    if label_names is None:
        raise ValueError("label_names is required (list of label column names).")

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(eval_data, batch_size, shuffle=False)

    # Try to get sizes robustly
    train_size = len(train_data)
    eval_size = len(eval_data)

    train_losses, val_losses, f1_hist, acc_hist = [], [], [], []

    best_score = np.inf if monitor != "f1" else -1

    for epoch in range(epochs):
        model.train()
        running_train = 0.0

        for X, Y in tqdm(train_loader, desc=f"Training {epoch+1}/{epochs}"):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)  # [B,K]
            loss = criterion(logits, Y)

            # ---- Optional eXBL explanation loss term ----
            if (
                cammer is not None
                and expl_triplet_loss is not None
                and alpha is not None
                and alpha > 0
                and C_good is not None
                and C_bad is not None
            ):
                B, K = logits.shape
                if img_size is None:
                    raise ValueError("img_size must be set when using cammer/eXBL.")
                # CAM tensor (store only what we compute)
                x_cam = torch.zeros(
                    (B, K, img_size, img_size), device=device, dtype=X.dtype
                )

                # decide which labels to compute CAM for
                if cam_use == "all":
                    label_mask = torch.ones((B, K), dtype=torch.bool, device=device)

                elif cam_use == "gt_pos":
                    label_mask = Y > 0.5

                elif cam_use == "pred_pos":
                    with torch.no_grad():
                        label_mask = torch.sigmoid(logits) > cam_threshold

                elif cam_use == "topk":
                    with torch.no_grad():
                        top_idx = torch.topk(
                            logits, k=min(topk, K), dim=1
                        ).indices  # [B,topk]
                        label_mask = torch.zeros(
                            (B, K), dtype=torch.bool, device=device
                        )
                        label_mask.scatter_(1, top_idx, True)
                else:
                    raise ValueError(f"Unknown cam_use='{cam_use}'")

                # compute CAMs (per-label backward). This is expensive if many labels are selected.
                # We loop labels (K) and backprop only on the subset of samples that need that label.
                for k in range(K):
                    sel = label_mask[:, k]
                    if not sel.any():
                        continue

                    model.zero_grad(set_to_none=True)
                    score = logits[sel, k].sum()
                    score.backward(retain_graph=True)

                    cam_k = cammer.compute_cam()
                    # expect cam_k: torch [B,H,W] on device
                    if not torch.is_tensor(cam_k):
                        raise TypeError(
                            "cammer.compute_cam() must return a torch.Tensor"
                        )
                    if cam_k.device != X.device:
                        cam_k = cam_k.to(device)

                    x_cam[:, k] = cam_k

                # add expl loss (your function should implement Eq. 9 behavior) :contentReference[oaicite:2]{index=2}
                loss = loss + alpha * expl_triplet_loss(X, x_cam, C_good, C_bad)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_train += loss.item() * X.size(0)

        train_loss = running_train / max(train_size, 1)
        train_losses.append(train_loss)

        # ---- Validation ----
        model.eval()
        running_val = 0.0
        preds = []

        with torch.no_grad():
            for X, Y in tqdm(val_loader, desc="Validation"):
                X = X.to(device)
                Y = Y.to(device)

                logits = model(X)
                loss = criterion(logits, Y)
                running_val += loss.item() * X.size(0)

                prob = torch.sigmoid(logits)
                preds.extend((prob > 0.5).cpu().long().numpy())

        preds = np.stack(preds)
        val_loss = running_val / max(eval_size, 1)
        val_losses.append(val_loss)

        # Ground truth array: prefer Y from dataset if available in the way you already do
        y_true = eval_data.data[label_names].to_numpy()
        val_acc = accuracy_score(y_true, preds)
        val_f1 = f1_score(y_true, preds, average="macro")
        acc_hist.append(val_acc)
        f1_hist.append(val_f1)

        print(
            f"Epoch: {epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} "
            f"- Val Acc: {val_acc:.4f} - Val F1(macro): {val_f1:.4f}"
        )

        # ---- checkpointing ----
        improved = False
        if monitor == "f1":
            if val_f1 > best_score:
                best_score = val_f1
                improved = True
        elif monitor == "loss":
            if val_loss < best_score:
                best_score = val_loss
                improved = True

        if improved and save_as is not None:
            print("Model improved! Saving checkpoint.")
            torch.save(model.state_dict(), save_as)

        if stepLR is not None:
            stepLR.step()

    if monitor is None and save_as is not None:
        torch.save(model.state_dict(), save_as)

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "f1": f1_hist,
        "accuracy": acc_hist,
        "epochs": epochs,
    }


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def compute_cam(self, out_size=(IMG_SIZE, IMG_SIZE), eps=1e-8):
        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "No activations/gradients captured. "
                "Make sure you ran a forward pass and backward() on a class score "
                "with gradients enabled before calling compute_cam()."
            )

        acts = self.activations
        grads = self.gradients

        weights = grads.mean(dim=(2, 3))

        cam = torch.relu((weights[:, :, None, None] * acts).sum(dim=1))

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + eps)

        if out_size is not None:
            cam = F.interpolate(
                cam[:, None, :, :],  # [B, 1, H, W]
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )[:, 0, :, :]

        return cam

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def make_circle_mask(batch_size, h, w, radius_ratio=0.48, device="cuda"):
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * radius_ratio)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r**2).float()  # [H,W]
    mask = mask[None, None, :, :].repeat(batch_size, 1, 1, 1)  # [B,1,H,W]
    return mask.to(device)


def activation_recall(cam_b3hw, mask_b1hw, eps=1e-8):
    # AR = sum(cam * M) / sum(M)
    num = (cam_b3hw * mask_b1hw).sum(dim=(2, 3))
    den = mask_b1hw.sum(dim=(2, 3)).clamp_min(eps)
    return (num / den).squeeze(1)  # [B,3]


@torch.no_grad()
def select_exemplars(images, cams, radius_ratio=0.48):
    """
    images: [B,3,H,W]
    cams: [B,K,H,W]

    return shape: [3,H,W]
    """
    model.eval()

    mask = make_circle_mask(images.shape[0], IMG_SIZE, IMG_SIZE, radius_ratio)

    ar = activation_recall(cams, mask)

    ar_max, idx_max = ar.max(dim=0)
    ar_min, idx_min = ar.min(
        dim=0
    )  # (3,): one for each class - index in batch dimension

    rng = torch.arange(cams.shape[1])

    C_good = images[idx_max] * cams[idx_max, rng].unsqueeze(1)  # [K,3,H,W]
    C_bad = images[idx_min] * cams[idx_min, rng].unsqueeze(1)  # [K,3,H,W]

    assert C_good is not None and C_bad is not None
    model.train()
    return C_good, C_bad, ar_max, ar_min


# 1. Loading unrefined model
checkpoints_dir = "trained_models/"
efficient_net_dir = "pretrained_backbone/ckpt_efficientnet_ep50.pt"
model = Classifier(backbone="efficientnet", dir=efficient_net_dir).to(device)
model.load_state_dict(torch.load(checkpoints_dir + "task1_efficient.pt"))
print("Model laoded successfully.")

# 2. Selecting exemplars in val pool
loader = DataLoader(val, BATCH, shuffle=True)

cammer = GradCAM(model, model.model.features[-1])

X, cams, Y, preds = eval_model(model, train, cam=cammer, shuffle=True)
X = normalize_image(X).to(device)

C_good, C_bad, ar_max, ar_min = select_exemplars(X, cams)
print("Selected exemplars AR:", ar_max, ar_min)

# 3. Training with the expl loss
# Full Fine tuning
for layer in model.parameters():
    layer.requires_grad = True

alpha = 0.2  # best
train_loader = DataLoader(train, BATCH, shuffle=True)
bce_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.model.parameters(), lr=1e-3, weight_decay=1e-4
)  # lr=1e-3
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

result = train_model(
    model,
    train,
    val,
    optimizer=optimizer,
    criterion=bce_loss,
    epochs=20,
    stepLR=scheduler,
    device=device,
    label_names=label_names,
    batch_size=BATCH,
    cammer=cammer,
    img_size=IMG_SIZE,
    alpha=alpha,
    save_as=checkpoints_dir + "task4_gradcam_efficientnet.pt",
    expl_triplet_loss=expl_triplet_loss,
    C_good=C_good,
    C_bad=C_bad,
    cam_use="gt_pos",  # recommended for multi-label
)
training_graphs(result, "task4/efficientnet_gradcam_full_tuning")

model.load_state_dict(torch.load(checkpoints_dir + "task4_gradcam_efficientnet.pt"))
eval_model(
    model, offsite_test, report_dir="task4/efficientnet_gradcam_report_full_tuning.txt"
)
eval_model(model, onsite_test, "task4/efficientnet_gradcam_submission_full.csv")

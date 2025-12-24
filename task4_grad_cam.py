import os

import cv2
from utils.losses import *
from utils.train_eval import *


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

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


efficientnet_dir = "pretrained_backbone/ckpt_efficientnet_ep50.pt"
model = Classifier(backbone="efficientnet", dir=efficientnet_dir).to(device)
checkpoints_dir = "trained_models/"

print("Efficientnet")
model.load_state_dict(torch.load(checkpoints_dir + "task1_efficient.pt"))
cam = GradCAM(model, model.model.features[-1])
images, cams, targets, preds = eval_model(
    model, offsite_test, cam=cam, cam_max_batches=100, shuffle=True
)

# overlay perclass cams on images and save
output_dir = "task4"
os.makedirs(output_dir, exist_ok=True)
print(images.shape, type(cams[0]), targets.shape, preds.shape)
for i in range(len(images)):
    img = (np.transpose(images[i], (1, 2, 0)) * 255).astype(np.uint8)
    target = targets[i]
    pred = preds[i]
    for class_idx in range(len(label_names)):
        if target[class_idx] == 1 or pred[class_idx] == 1:
            cam_map = cams[class_idx][i]
            heatmap = cv2.applyColorMap(
                (cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"img{i}_class{class_idx}_t{target[class_idx]}_p{pred[class_idx]}.png",
                ),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            cams[class_idx].append(cam_map)

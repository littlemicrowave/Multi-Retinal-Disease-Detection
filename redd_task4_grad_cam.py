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
output_dir = "task4/grad_cam_outputs/"
os.makedirs(output_dir, exist_ok=True)
for i in range(len(images)):
    img = normalize_image(np.transpose(images[i], (1, 2, 0)))
    for label in np.where(targets[i] == 1)[0]:
        print(label, type(label))
        cm = normalize_image(cams[int(label)][i])
        plt.imshow(img)
        plt.imshow(cm, alpha=0.55)
        plt.title(f"target: {targets[i].astype(np.int8)} \n preds: {preds[i]}")
        correct = "true" if targets[i][label] == preds[i][label] else "false"
        plt.savefig(f"{output_dir}{i}_{label_names[int(label)]}_{correct}.png")
        plt.clf()

import random

import numpy as np
import torch
from PIL import Image


def normalize_image(img, eps=1e-8):
    """
    Min–max normalize image(s) to [0,1].

    Supports:
      - [C, H, W]
      - [B, C, H, W]

    Normalization is done PER IMAGE (not across the batch).
    """
    if not torch.is_tensor(img):
        img = torch.tensor(img)

    if img.dim() == 3:
        # [C,H,W] -> [1,C,H,W]
        img = img.unsqueeze(0)
        squeeze_back = True
    elif img.dim() == 4:
        squeeze_back = False
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {img.shape}")

    B = img.size(0)

    # flatten spatial+channel dims
    img_flat = img.view(B, -1)

    min_val = img_flat.min(dim=1)[0].view(B, 1, 1, 1)
    max_val = img_flat.max(dim=1)[0].view(B, 1, 1, 1)

    img_norm = (img - min_val) / (max_val - min_val + eps)

    if squeeze_back:
        img_norm = img_norm.squeeze(0)

    return img_norm


class RandomizeOutsideCircle:
    def __init__(self, p=1.0, radius_ratio=0.48, mode="noise"):
        """
        p: probability of applying (use <1.0 if you want it stochastic)
        radius_ratio: circle radius relative to image size
        mode: "noise" or "zero"
        """
        self.p = p
        self.radius_ratio = radius_ratio
        self.mode = mode

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # img is a PIL image here
        w, h = img.size
        cx, cy = w // 2, h // 2
        r = int(min(w, h) * self.radius_ratio)

        img_np = np.array(img).astype(np.float32)

        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        mask = mask[..., None]  # shape (H, W, 1)

        if self.mode == "noise":
            outside = np.random.uniform(0, 255, size=img_np.shape)
        else:  # "zero"
            outside = np.zeros_like(img_np)

        img_np = img_np * mask + outside * (1 - mask)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)


class RandomizeOutsideCircle:
    def __init__(self, p=1.0, radius_ratio=0.48, mode="noise"):
        """
        p: probability of applying (use <1.0 if you want it stochastic)
        radius_ratio: circle radius relative to image size
        mode: "noise" or "zero"
        """
        self.p = p
        self.radius_ratio = radius_ratio
        self.mode = mode

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # img is a PIL image here
        w, h = img.size
        cx, cy = w // 2, h // 2
        r = int(min(w, h) * self.radius_ratio)

        img_np = np.array(img).astype(np.float32)

        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        mask = mask[..., None]  # shape (H, W, 1)

        if self.mode == "noise":
            outside = np.random.uniform(0, 255, size=img_np.shape)
        else:  # "zero"
            outside = np.zeros_like(img_np)

        img_np = img_np * mask + outside * (1 - mask)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)

import random

import numpy as np
from PIL import Image


def normalize_image(img):
    mini = img.min()
    maxi = img.max()
    return (img - mini) / (maxi - mini)


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

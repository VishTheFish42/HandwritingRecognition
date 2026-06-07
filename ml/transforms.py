"""
Augmentation transforms for handwriting line images.

All transforms accept and return PIL Images so they compose naturally with
torchvision.transforms. Each has a `p` parameter (application probability).
"""

import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class RandomRotation:
    """Rotate by a uniform random angle within [-degrees, +degrees].

    Background fill is white (255) to match handwriting-on-white images.
    """

    def __init__(self, degrees: float = 5.0, p: float = 0.5) -> None:
        self.degrees = degrees
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)


class ElasticDistortion:
    """Elastic deformation via random smooth displacement fields (Simard et al., 2003).

    alpha controls displacement magnitude; sigma controls field smoothness.
    Larger sigma = smoother, more global warp. Smaller = more local jitter.
    """

    def __init__(
        self, alpha: float = 34.0, sigma: float = 4.0, p: float = 0.5
    ) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr = np.array(img)
        h, w = arr.shape[:2]

        noise_x = (np.random.rand(h, w).astype(np.float32) * 2 - 1)
        noise_y = (np.random.rand(h, w).astype(np.float32) * 2 - 1)

        dx = cv2.GaussianBlur(noise_x, ksize=(0, 0), sigmaX=self.sigma) * self.alpha
        dy = cv2.GaussianBlur(noise_y, ksize=(0, 0), sigmaX=self.sigma) * self.alpha

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                      np.arange(h, dtype=np.float32))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        distorted = cv2.remap(
            arr, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(distorted)


class GaussianNoise:
    """Add zero-mean Gaussian noise with the given standard deviation (in [0, 1] scale)."""

    def __init__(self, std: float = 0.05, p: float = 0.5) -> None:
        self.std = std
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0.0, self.std, arr.shape).astype(np.float32), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


class BrightnessJitter:
    """Independently jitter brightness and contrast by a random factor in [1-delta, 1+delta]."""

    def __init__(
        self, brightness: float = 0.3, contrast: float = 0.3, p: float = 0.5
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        if self.brightness > 0 and random.random() < 0.5:
            factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0 and random.random() < 0.5:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img

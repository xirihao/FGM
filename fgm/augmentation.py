"""The 64-view augmentation protocol used by the image experiments.

AugMix operations are in third_party/augmix_ops.py (Google, Apache-2.0).
Random draws and transform order follow the evaluated augmentation protocol.
"""
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from third_party.augmix_ops import augmentations


def get_preaugment():
    return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])


def augmix(image, preprocess, aug_list, severity=1):
    x_orig = get_preaugment()(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    weights = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    mixture = np.float32(np.random.beta(1.0, 1.0))
    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += weights[i] * preprocess(x_aug)
    return mixture * x_processed + (1 - mixture) * mix


class AugMixAugmenter:
    def __init__(self, base_transform, preprocess, n_views=63, augmix=True, severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = augmentations if augmix else []
        self.severity = severity

    def __call__(self, image):
        center_view = self.preprocess(self.base_transform(image))
        views = [augmix(image, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [center_view] + views


def get_ood_preprocess():
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

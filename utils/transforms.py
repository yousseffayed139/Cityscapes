import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from typing import Dict

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        for key in sample.keys():
            interpolation = Image.NEAREST if key == 'mask' else Image.BILINEAR
            sample[key] = TF.resize(sample[key], self.size, interpolation=interpolation)
        return sample

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        angle = np.random.uniform(-self.degrees, self.degrees)
        for key in sample.keys():
            sample[key] = TF.rotate(sample[key], angle)
        return sample

class ToTensor:
    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, torch.Tensor]:
        sample['image'] = TF.to_tensor(sample['image'])
        sample['mask'] = torch.tensor(np.array(sample['mask']), dtype=torch.long)
        return sample

class NormalizeImage:
    def __init__(self, mean, std, clip_limit=1.5, tile_grid_size=(16, 16)):
        self.mean = mean
        self.std = std
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = sample['image'].numpy()
        image = np.moveaxis(image, 0, -1)
        clahe_channels = [self.clahe.apply((image[:, :, i] * 255).astype(np.uint8)) for i in range(image.shape[2])]
        image_clahe = np.stack(clahe_channels, axis=-1) / 255.0
        image_clahe = np.moveaxis(image_clahe, -1, 0)
        image_clahe = torch.tensor(image_clahe).float()
        image_clahe = TF.normalize(image_clahe, mean=self.mean, std=self.std)
        sample['image'] = image_clahe
        return sample

class NormalizeLabels:
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = sample['mask'].numpy()
        normalized_mask = np.copy(mask)
        valid_range = range(0, 20)
        normalized_mask[(normalized_mask >= 0) & (normalized_mask < 20)] += 1
        normalized_mask[~np.isin(normalized_mask, list(valid_range) + [255])] = 0
        normalized_mask[normalized_mask == 255] = 0
        sample['mask'] = torch.tensor(normalized_mask, dtype=torch.long)
        return sample

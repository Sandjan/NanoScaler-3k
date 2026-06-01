import hashlib
import math
import os
import pickle
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RandomNativePatch(nn.Module):
    patch_size: List[int]
    log_min_scale: float
    log_max_scale: float

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 80,
        scale: Tuple[float, float] = (1.7, 25.0),
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        else:
            self.patch_size = [patch_size[0], patch_size[1]]
        self.log_min_scale = math.log(scale[0])
        self.log_max_scale = math.log(scale[1])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Float-Tensor [C, H, W]
        Returns:
            HR-Patch [C, patch_size[0], patch_size[1]]
        """
        h: int = img.shape[-2]
        w: int = img.shape[-1]

        rng = torch.rand(3, dtype=torch.float32, device=img.device)

        # Log-uniform scale
        s = math.exp(
            self.log_min_scale
            + (self.log_max_scale - self.log_min_scale) * rng[0].item()
        )

        # Snap crop size to image edges
        crop_h: int = min(int(self.patch_size[0] * s), h)
        crop_w: int = min(int(self.patch_size[1] * s), w)

        # Random crop position within the valid range
        max_top: int = max(h - crop_h, 0)
        max_left: int = max(w - crop_w, 0)
        top: int = min(int(rng[1].item() * float(max_top + 1)), max_top)
        left: int = min(int(rng[2].item() * float(max_left + 1)), max_left)

        # Zero-copy view + area resample
        return F.interpolate(
            img[:, top : top + crop_h, left : left + crop_w].unsqueeze(0),
            size=[self.patch_size[0], self.patch_size[1]],
            mode="area",
        ).squeeze(0)


# ------------------- Dataset -------------------
class RAMImageDataset(Dataset):
    def __init__(self, folder_paths, hr_size=200, preload_to_ram=True):
        self.hr_size = hr_size
        self.image_paths = [
            os.path.join(path, f)
            for path in folder_paths
            for f in os.listdir(path)
            if f.endswith(("jpg", "png"))
        ]

        self.images = []
        if preload_to_ram:
            cache_path = self._get_cache_path(folder_paths[0])
            if os.path.exists(cache_path):
                print("Loading images from cache...")
                with open(cache_path, "rb") as f:
                    self.images = pickle.load(f)
            else:
                print(f"Loading {len(self.image_paths)} Images to RAM...")
                for p in self.image_paths:
                    self.images.append(Image.open(p).convert("RGB"))
                print("Saving Cache...")
                with open(cache_path, "wb") as f:
                    pickle.dump(self.images, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.transform_hr = torch.jit.script(RandomNativePatch(hr_size, (1.7, 20)))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def _get_cache_path(self, folder_path):
        folder_hash = hashlib.md5(folder_path.encode()).hexdigest()[:8]
        cache_dir = os.path.join(folder_path, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"images_{folder_hash}.pkl")

    def _load(self, idx):
        return (
            self.images[idx]
            if self.images
            else Image.open(self.image_paths[idx]).convert("RGB")
        )

    def __getitem__(self, idx):
        img = self._load(idx)
        hr = self.transform_hr(self.to_tensor(img))
        return hr

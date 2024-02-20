from pathlib import Path

import numpy as np
from pytorchvideo.transforms import ShortSideScale
import torch
import torchvision
from PIL import Image
from decord import VideoReader


class VideoReaderFromImages:
    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self.images_list = sorted(self.path.iterdir())
        self._num_frame = len(self.images_list)

    def __len__(self):
        return self._num_frame

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(len(self))))
        image = torchvision.io.read_image(str(self.path / self.images_list[idx]))
        image = image.permute(1, 2, 0)
        return image

    def _validate_indices(self, indices):
        """Validate int64 integers and convert negative integers to positive by backward search"""
        indices = np.array(indices, dtype=np.int64)
        # process negative indices
        indices[indices < 0] += self._num_frame
        if not (indices >= 0).all():
            raise IndexError(
                'Invalid negative indices: {}'.format(indices[indices < 0] + self._num_frame))
        if not (indices < self._num_frame).all():
            raise IndexError('Out of bound indices: {}'.format(indices[indices >= self._num_frame]))
        return indices

    def get_avg_fps(self):
        return 10  # TODO: verify this

    def get_batch(self, indices):
        indices = self._validate_indices(indices)
        return torch.cat([self[idx][None, ...] for idx in indices], dim=0)

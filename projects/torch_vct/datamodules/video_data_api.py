# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, NamedTuple, Optional, Union

import torch
from torch import Tensor


class VideoDataset(NamedTuple):
    video_tensor: Tensor  # [B, T, 3, H, W]

    @classmethod
    def make_random_video(cls, B: int = 2, T: int = 5, H: int = 64, W: int = 64):
        """Create a random video"""
        base_shape = (B, T, 3, H, W)
        video_tensor = torch.randn(base_shape)
        return cls(video_tensor)

    @property
    def shape(self):
        """Return the number of frames in the video."""
        return self.video_tensor.shape

    @property
    def num_frames(self):
        """Return the number of frames in the video, T"""
        return self.video_tensor.shape[1]

    @property
    def batch_size(self):
        """Return the batch size, B"""
        return self.video_tensor.shape[0]

    @property
    def spatial_shape(self):
        """Return (H, W)"""
        return self.shape[-2], self.shape[-1]

    def validate_shape(self):
        """Raise ValueError if we have invalid shapes."""
        if len(self.shape) != 5 or self.shape[2] != 3:
            raise ValueError(
                f"Invalid shape! Expected [B, T, 3, H, W], got {self.shape}"
            )


class KineticsDataset(VideoDataset):
    video_name: Optional[List[str]] = None
    label: Optional[Tensor] = None
    video_index: Optional[Tensor] = None
    clip_index: Optional[Tensor] = None
    aug_index: Optional[Tensor] = None


class VimeoDataset(VideoDataset):
    label: Optional[Tensor] = None


VideoData = Union[VideoDataset, KineticsDataset, VimeoDataset]


class Scenes(NamedTuple):
    """Represent a batch of latents/scenes"""

    tensor: Tensor  # [B, T, C, H, W]

    @property
    def shape(self):
        """Return the shape"""
        return self.tensor.shape

    @property
    def batch_size(self):
        """Return the batch size, B"""
        return self.tensor.shape[0]

    @property
    def num_scenes(self):
        """Return number of scenes/latents"""
        return self.tensor.shape[1]

    @property
    def spatial_shape(self):
        """Return spatial dimensions (H, W)"""
        H, W = self.tensor.shape[-2], self.tensor.shape[-1]
        return H, W

    def validate_shapes(self):
        """Raise ValueError if we have invalid shapes."""
        # Accept any floating-point dtype so mixed-precision (bf16 / fp16
        # autocast) is allowed. The original code hard-required float32,
        # which broke torch.amp.autocast since the latent tensor coming
        # out of analysis_transform is downcast to bf16 inside the autocast
        # region. Shape correctness is what this validator actually cares
        # about.
        if not self.tensor.is_floating_point():
            raise ValueError(f"Expected floating-point dtype, got {self.tensor.dtype}")
        if self.tensor.dim() != 5:
            raise ValueError(f"Expected (B, T, C, H, W), got {self.tensor.shape}")

    def get_scenes_iter(self) -> List[Tensor]:
        """Iterat over T, return batches of single scenes/latents"""
        self.validate_shapes()
        scenes = []
        for i in range(self.num_scenes):
            scenes.append(self.tensor[:, i, ...])
        return scenes

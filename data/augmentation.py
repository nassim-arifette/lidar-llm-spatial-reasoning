import numpy as np
import torch
from typing import Dict, List, Union, Tuple, Callable
from abc import ABC, abstractmethod
import random


class BaseAugmentation(ABC):
    @abstractmethod
    def __call__(self, data):
        pass


class PointCloudAugmentation(BaseAugmentation):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        for aug in self.augmentations:
            data = aug(data)
        return data


class RandomRotation(BaseAugmentation):
    def __init__(self, angle_range=(-np.pi, np.pi)):
        self.angle_range = angle_range

    def __call__(self, data):
        points = data['points']

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.numpy()
        else:
            points_np = points

        angle = random.uniform(*self.angle_range)

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])

        xyz = points_np[:, :3]
        rotated_xyz = np.dot(xyz, rotation_matrix.T)

        points_np[:, :3] = rotated_xyz

        if is_tensor:
            data['points'] = torch.from_numpy(points_np).to(points.dtype).to(points.device)
        else:
            data['points'] = points_np

        return data


class RandomJitter(BaseAugmentation):
    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

    def __call__(self, data):
        points = data['points']

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.numpy()
        else:
            points_np = points

        jitter = np.clip(np.random.normal(0, self.std, size=points_np[:, :3].shape), -self.clip, self.clip)

        points_np[:, :3] += jitter

        if is_tensor:
            data['points'] = torch.from_numpy(points_np).to(points.dtype).to(points.device)
        else:
            data['points'] = points_np

        return data


class RandomScale(BaseAugmentation):
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, data):
        points = data['points']

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.numpy()
        else:
            points_np = points

        scale = random.uniform(*self.scale_range)

        points_np[:, :3] *= scale

        if is_tensor:
            data['points'] = torch.from_numpy(points_np).to(points.dtype).to(points.device)
        else:
            data['points'] = points_np

        return data


class RandomTranslation(BaseAugmentation):
    def __init__(self, translation_range=(-0.2, 0.2)):
        self.translation_range = translation_range

    def __call__(self, data):
        points = data['points']

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.numpy()
        else:
            points_np = points

        tx = random.uniform(*self.translation_range)
        ty = random.uniform(*self.translation_range)
        tz = random.uniform(*self.translation_range)

        points_np[:, 0] += tx
        points_np[:, 1] += ty
        points_np[:, 2] += tz

        if is_tensor:
            data['points'] = torch.from_numpy(points_np).to(points.dtype).to(points.device)
        else:
            data['points'] = points_np

        return data


class RandomDropout(BaseAugmentation):
    def __init__(self, dropout_ratio=0.2):
        self.dropout_ratio = dropout_ratio

    def __call__(self, data):
        points = data['points']

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.numpy()
        else:
            points_np = points

        n_points = points_np.shape[0]
        n_keep = int(n_points * (1 - self.dropout_ratio))

        keep_indices = np.random.choice(n_points, n_keep, replace=False)
        points_np = points_np[keep_indices]

        if is_tensor:
            data['points'] = torch.from_numpy(points_np).to(points.dtype).to(points.device)
        else:
            data['points'] = points_np

        return data


class RandomFlip(BaseAugmentation):
    def __init__(self, flip_x=True, flip_y=True, flip_z=False, p=0.5):
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
        self.p = p

    def __call__(self, data):
        points = data['points']

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.numpy()
        else:
            points_np = points

        if self.flip_x and random.random() < self.p:
            points_np[:, 0] = -points_np[:, 0]

        if self.flip_y and random.random() < self.p:
            points_np[:, 1] = -points_np[:, 1]

        if self.flip_z and random.random() < self.p:
            points_np[:, 2] = -points_np[:, 2]

        if is_tensor:
            data['points'] = torch.from_numpy(points_np).to(points.dtype).to(points.device)
        else:
            data['points'] = points_np

        return data


class ElasticDistortion(BaseAugmentation):
    def __init__(self, granularity=0.2, magnitude=0.8):
        self.granularity = granularity
        self.magnitude = magnitude

    def __call__(self, data):
        points = data['points']

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.numpy()
        else:
            points_np = points

        min_coords = points_np[:, :3].min(axis=0)
        max_coords = points_np[:, :3].max(axis=0)
        dimensions = max_coords - min_coords

        resolution = np.ceil(dimensions / self.granularity).astype(int)
        resolution = np.maximum(resolution, 2)

        noise_dim = [
            np.random.randn(*resolution) * self.magnitude for _ in range(3)
        ]

        coords_normalized = (points_np[:, :3] - min_coords) / dimensions

        indices = coords_normalized * (resolution - 1)

        displacements = np.zeros_like(points_np[:, :3])
        for dim in range(3):
            indices_floor = np.floor(indices).astype(int)
            indices_ceil = np.minimum(indices_floor + 1, np.array(resolution) - 1)
            fraction = indices - indices_floor

            # Interpoler les valeurs de bruit
            for i in range(len(points_np)):
                x, y, z = indices_floor[i]
                x1, y1, z1 = indices_ceil[i]
                fx, fy, fz = fraction[i]

                # Interpolation trilinéaire
                c000 = noise_dim[dim][x, y, z]
                c001 = noise_dim[dim][x, y, z1]
                c010 = noise_dim[dim][x, y1, z]
                c011 = noise_dim[dim][x, y1, z1]
                c100 = noise_dim[dim][x1, y, z]
                c101 = noise_dim[dim][x1, y, z1]
                c110 = noise_dim[dim][x1, y1, z]
                c111 = noise_dim[dim][x1, y1, z1]

                c00 = c000 * (1 - fz) + c001 * fz
                c01 = c010 * (1 - fz) + c011 * fz
                c10 = c100 * (1 - fz) + c101 * fz
                c11 = c110 * (1 - fz) + c111 * fz

                c0 = c00 * (1 - fy) + c01 * fy
                c1 = c10 * (1 - fy) + c11 * fy

                displacements[i, dim] = c0 * (1 - fx) + c1 * fx

        points_np[:, :3] += displacements

        if is_tensor:
            data['points'] = torch.from_numpy(points_np).to(points.dtype).to(points.device)
        else:
            data['points'] = points_np

        return data


class Compose:
    def __init__(self, transforms: List[BaseAugmentation]):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def create_aug():
    elastic = ElasticDistortion()
    dropout = RandomDropout()
    jitter = RandomJitter()
    translation = RandomTranslation()
    scale = RandomScale()
    return Compose([elastic, dropout, jitter, translation, scale])

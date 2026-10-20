import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, List, Dict, Any, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json
from abc import ABC, abstractmethod



class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class ImageDataset(BaseDataset):
    def __init__(
        self,
        image_dir: str,
        annotations_file: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.image_dir = image_dir
        self.transform = transform

        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
            self.image_paths = [os.path.join(image_dir, filename) for filename in self.annotations["filename"]]
            self.labels = self.annotations["label"].values
        else:
            self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
            self.labels = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label

        return image

    def _load_image(self, path):
        from PIL import Image
        return Image.open(path).convert("RGB")



#############################################
# LidarPointCloudDataset for Raw LiDAR Data
#############################################
class LidarPointCloudDataset(Dataset):
    def __init__(self, root_dir, prompts=None, transform=None,
                 max_points=None, cache_data=False, answers=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_points = max_points
        self.cache_data = cache_data
        self.prompts = prompts or []
        self.answers = answers or []

        self.point_cloud_files = []
        for ext in ['.bin', '.ply', '.pcd', '.xyz', '.pts', '.npz']:
            self.point_cloud_files.extend(glob.glob(os.path.join(root_dir, f'*{ext}')))
        self.point_cloud_files.sort()

        if not self.prompts:
            self.prompts = [""] * len(self.point_cloud_files)

        if not self.answers:
            self.answers = [""] * len(self.point_cloud_files)

        self.data_cache = {} if cache_data else None

        print(f"Found {len(self.point_cloud_files)} point cloud files in {root_dir}")

    def _load_bin_file(self, file_path):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points

    def _load_ply_file(self, file_path):
        try:
            from plyfile import PlyData
            plydata = PlyData.read(file_path)
            data = plydata['vertex'].data
            x = data['x']
            y = data['y']
            z = data['z']
            if 'intensity' in data.dtype.names:
                intensity = data['intensity']
                points = np.vstack((x, y, z, intensity)).T
            else:
                points = np.vstack((x, y, z)).T
            return points
        except ImportError:
            print("La bibliothèque 'plyfile' est requise pour charger les fichiers PLY. Installez-la avec pip install plyfile.")
            raise

    def _load_pcd_file(self, file_path):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                points = np.hstack((points, colors))
            return points
        except ImportError:
            print("La bibliothèque 'open3d' est requise pour charger les fichiers PCD. Installez-la avec pip install open3d.")
            raise

    def _load_txt_file(self, file_path):
        points = np.loadtxt(file_path, delimiter=' ')
        return points

    def _load_npz_file(self, file_path):
        data = np.load(file_path)
        # Debug: Print available keys
        #print(f"Keys in {os.path.basename(file_path)}: {data.files}")
        if 'points' in data:
            points = data['points']
        elif 'xyz' in data:
            points = data['xyz']
        elif 'pointcloud' in data:
            points = data['pointcloud']
        else:
            for key in data.files:
                points = data[key]
                print(f"Using key '{key}' with shape {points.shape}")
                break
        return points

    def __len__(self):
        return len(self.point_cloud_files)

    def __getitem__(self, idx):
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]

        file_path = self.point_cloud_files[idx]
        prompt = self.prompts[idx]
        answer = self.answers[idx]
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.bin':
                point_cloud = self._load_bin_file(file_path)
            elif file_extension == '.ply':
                point_cloud = self._load_ply_file(file_path)
            elif file_extension == '.pcd':
                point_cloud = self._load_pcd_file(file_path)
            elif file_extension in ['.xyz', '.pts']:
                point_cloud = self._load_txt_file(file_path)
            elif file_extension == '.npz':
                point_cloud = self._load_npz_file(file_path)
            else:
                raise ValueError(f"Format de fichier non pris en charge: {file_extension}")

            if self.max_points is not None and point_cloud.shape[0] > self.max_points:
                indices = np.random.choice(point_cloud.shape[0], self.max_points, replace=False)
                point_cloud = point_cloud[indices]

            point_cloud_tensor = torch.from_numpy(point_cloud).float()
            if self.transform:
                point_cloud_tensor = self.transform(point_cloud_tensor)

            result = {
                'points': point_cloud_tensor,
                'prompt': prompt,
                'answer': answer,
                'file_path': file_path,
                'index': idx
            }
            if self.cache_data:
                self.data_cache[idx] = result
            return result
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return {
                'points': torch.zeros((1, 4)),
                'prompt': f"Error: {str(e)}",
                'answer': "",
                'file_path': file_path,
                'index': idx,
                'error': str(e)
            }

    def embed(self, model):
        """
        Optionally pre-compute embeddings for each point cloud using the provided model.
        This function checks if the tensor is transposed (i.e. shape[0] == 4) and transposes it if needed.
        """
        if model is None:
            print("No model provided for embedding, skipping embedding step.")
            return

        with torch.no_grad():
            for idx in tqdm(range(len(self)), desc="Preprocessing Data", leave=False):
                data = self[idx]
                point_cloud_tensor = data['points']
                # Check if the tensor is transposed (e.g., 4 x N instead of N x 4)
                if point_cloud_tensor.dim() == 2 and point_cloud_tensor.shape[0] == 4 and point_cloud_tensor.shape[1] != 4:
                    point_cloud_tensor = point_cloud_tensor.T
                embedded_points = model(point_cloud_tensor)
                data['points'] = embedded_points
                if self.cache_data:
                    self.data_cache[idx] = data


#############################################
# NuscenesFeatureDataset for Pre-extracted Features
#############################################
class NuscenesFeatureDataset(Dataset):
    def __init__(self, feature_dir: str, annotation_file: str, transform: Optional[Callable] = None):
        """
        Args:
            feature_dir: Directory containing pre-extracted feature files (e.g. .npy files).
            annotation_file: JSON file containing filtered annotations.
            transform: Optional transform to apply to the features.
        """
        self.feature_dir = feature_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.sample_tokens = [entry["sample_token"] for entry in self.annotations]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        sample_token = annotation["sample_token"]
        feature_path = os.path.join(self.feature_dir, f"{sample_token}.npy")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        feature = np.load(feature_path)
        feature = torch.from_numpy(feature).float()
        if self.transform:
            feature = self.transform(feature)
        return {
            "points": feature,
            "prompt": annotation.get("prompt", ""),
            "answer": annotation.get("answer", ""),
            "sample_token": sample_token,
            "class_label": annotation.get("class_label", None),
            "bev_box": annotation.get("bev_box", None)
        }

#############################################
# Custom Collate Function for Variable-Sized Data
#############################################
def collate_fn(batch):
    batch = [item for item in batch if 'error' not in item]
    if len(batch) == 0:
        return {
            'points': torch.zeros((0, 4)),
            'prompt': [],
            'answer': [],
            'file_path': [],
            'index': []
        }
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key == 'points':
            result[key] = [item[key] for item in batch]
        elif key in ['prompt', 'answer', 'file_path', 'index', 'sample_token']:
            result[key] = [item[key] for item in batch]
        else:
            try:
                result[key] = torch.stack([item[key] for item in batch])
            except Exception:
                result[key] = [item[key] for item in batch]
    return result

#############################################
# DataLoader Creation
#############################################
def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    return train_loader, val_loader, test_loader

#############################################
# Dataset Splitting: Create Train, Test, Val Datasets
#############################################
def create_train_test_val_datasets(
        model, 
        root_dir: str,
        num_samples: int = 50,
        max_points: int = 10000,
        seed: int = 42,
        use_preextracted: bool = False,
        feature_dir: Optional[str] = None,
        annotation_file: Optional[str] = None
    ):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(root_dir, exist_ok=True)

    # Check if the directory has train/val split folders
    train_folder = os.path.join(root_dir, "train")
    val_folder = os.path.join(root_dir, "val")
    if os.path.isdir(train_folder) and os.path.isdir(val_folder):
        print(f"Using provided split from {root_dir}:")
        print(f"  Training folder: {train_folder}")
        print(f"  Validation folder: {val_folder}")
        train_dataset = LidarPointCloudDataset(
            root_dir=train_folder,
            max_points=max_points,
            cache_data=True
        )
        val_dataset = LidarPointCloudDataset(
            root_dir=val_folder,
            max_points=max_points,
            cache_data=True
        )
        test_dataset = None
        train_dataset.embed(model)
        val_dataset.embed(model)
        return train_dataset, test_dataset, val_dataset

    else:
        # Check if there are any point cloud files in the root_dir
        existing_files = []
        for ext in ['.bin', '.ply', '.pcd', '.xyz', '.pts', '.npz']:
            existing_files.extend(glob.glob(os.path.join(root_dir, f'*{ext}')))
        existing_files.sort()
        if len(existing_files) > 0:
            print(f"Using existing dataset from {root_dir} with {len(existing_files)} files.")
            full_dataset = LidarPointCloudDataset(
                root_dir=root_dir,
                max_points=max_points,
                cache_data=True
            )
        else:
            print(f"No point cloud files found in {root_dir}. Generating a random dataset with {num_samples} samples.")
            example_prompts = [
                "Identifier les obstacles sur la route",
                "Détecter les piétons dans la scène",
                "Analyser la structure de la route",
                "Segmenter les véhicules environnants",
                "Mesurer la distance aux objets proches",
                "Analyser la densité du trafic",
                "Repérer les panneaux de signalisation",
                "Détecter les feux de circulation",
                "Identifier les bâtiments",
                "Tracer le contour de la chaussée"
            ]
            possible_answers = [
                "Utilisation de capteurs LIDAR et de caméras pour détecter les objets statiques et dynamiques sur la voie.",
                "Application de modèles de vision par ordinateur pour identifier et suivre les piétons en mouvement.",
                "Analyse des marquages au sol et des bordures pour comprendre la géométrie de la route.",
                "Utilisation de réseaux de neurones convolutifs pour distinguer et segmenter les véhicules proches.",
                "Utilisation de capteurs ultrasoniques ou LIDAR pour estimer la distance aux objets environnants.",
                "Calcul du nombre de véhicules par unité de temps pour évaluer la congestion routière.",
                "Détection et reconnaissance des panneaux à l'aide de techniques de traitement d'image et d'apprentissage profond.",
                "Identification des feux de signalisation et de leur état (vert, orange, rouge) en temps réel.",
                "Utilisation de données cartographiques et de capteurs pour localiser et identifier les structures bâties.",
                "Utilisation de techniques de segmentation d'image pour délimiter les bords de la route."
            ]
            prompts = [random.choice(example_prompts) for _ in range(num_samples)]
            answers = [random.choice(possible_answers) for _ in range(num_samples)]
            for i in range(num_samples):
                num_points = random.randint(max_points // 2, max_points)
                points = np.random.rand(num_points, 4)
                file_path = os.path.join(root_dir, f"point_cloud_{i:05d}.bin")
                # Ensure the directory for file_path exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                points.astype(np.float32).tofile(file_path)
            full_dataset = LidarPointCloudDataset(
                root_dir=root_dir,
                prompts=prompts,
                answers=answers,
                max_points=max_points,
                cache_data=True
            )
        full_dataset.embed(model)

        dataset_length = len(full_dataset)
        train_size = int(0.8 * dataset_length)
        test_size = int(0.1 * dataset_length)
        val_size = dataset_length - train_size - test_size

        train_dataset, test_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, test_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        return train_dataset, test_dataset, val_dataset

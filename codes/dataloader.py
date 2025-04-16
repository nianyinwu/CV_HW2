""" Dataloader utilities for training, validation, and testing. """

import os
from argparse import Namespace
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes


def data_transform(mode):
    '''
    Transform for training and validation datasets.
    '''
    if mode == 'train':
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            v2.RandomAdjustSharpness(sharpness_factor=10, p=0.8),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

class Dataset(CocoDetection):
    '''
    Dataset for training and validation.
    '''

    def __init__(self, img_dir, ann_file, transforms=None):
        super().__init__(img_dir, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)

        boxes = []
        labels = []

        for t in targets:
            x, y, w, h = t['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(t['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

         # (W, H) → (H, W)
        image_size = image.size[::-1]
        boxes = BoundingBoxes(boxes, format="XYXY", canvas_size=image_size)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(self.ids[idx])
        }

        if self._transforms:
            image, target = self._transforms(image, target)

        return image, target

class TestDataset:
    '''
    Dataset for inference.
    '''
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.filename = sorted(os.listdir(self.root))

    def __len__(self) -> int:
        return len(self.filename)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root, self.filename[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, self.filename[idx]

def collate_fn(batch):
    '''
    Collate function.
    '''
    return tuple(zip(*batch))

def dataloader(args: Namespace, mode: str) -> DataLoader:
    """
    Create dataloader based on the mode: train, val, or test.

    Args:
        args (Namespace): Command-line arguments containing data_path and batch_size.
        mode (str): Mode of the data loader ('train', 'val', 'test').

    Returns:
        DataLoader: PyTorch DataLoader for the corresponding dataset.
    """

    dataset = None
    shuffle = False

    if mode in ['train', 'valid']:
        img_dir = os.path.join(args.data_path, mode)
        ann_file = os.path.join(args.data_path, f'{mode}.json')
        transform = data_transform(mode)
        dataset = Dataset(img_dir, ann_file, transforms=transform)
        if mode == 'train':
            shuffle = True
    elif mode == 'test':
        img_dir = os.path.join(args.data_path, mode)
        transform = data_transform(mode)
        dataset = TestDataset(img_dir, transform=transform)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=6,
        pin_memory=True,
        collate_fn=collate_fn
    )

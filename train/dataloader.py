import os
from tqdm import tqdm

import cv2
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# MEAN = [0.4488, 0.4371, 0.4040]
# STD = [1., 1., 1.]

MEAN = [0.5312, 0.4265, 0.3753]
STD = [0.2873, 0.2555, 0.2496]

class FaceDataset(Dataset):
    def __init__(self, root_dirs, p):
        super(FaceDataset, self).__init__()
        self.path_list, self.label_list = self.data_scan(root_dirs)
        self.num_classes = len(np.unique(np.array(self.label_list)))
        # Transform
        augment = A.Compose([
            A.RandomRotate90(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.Blur(p=p),
            A.MedianBlur(p=p),
            A.CLAHE(p=p),
            A.RandomBrightnessContrast(p=p),
            A.RandomGamma(p=p),
            A.ImageCompression(quality_lower=20, quality_upper=100, always_apply=True)
            ])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.transform = lambda x: transform(augment(image=x)["image"])

    def data_scan(self, root_dirs):
        path_list = []
        label_list = []
        idx = 0
        for dir in root_dirs:
            id_names = sorted(os.listdir(dir))
            for id_n in tqdm(id_names):
                img_names = sorted(os.listdir(os.path.join(dir, id_n)))
                for n in img_names:
                    path_list.append(os.path.join(dir, id_n, n))
                    label_list.append(idx)
                idx += 1
        return path_list, label_list

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        path, lb = self.path_list[index], self.label_list[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, lb
    

class ParallelDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(self.dataset)
        self.total_size = self.num_samples
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def dataloader_generation(data_dirs, batch_size, num_worker):
    ds = FaceDataset(data_dirs, p=0.5)
    loader = DataLoader(ds,
                        batch_size,
                        shuffle=True,
                        num_workers=num_worker,
                        pin_memory=True,
                        drop_last=False)
    return loader


def distdataloader_generation(data_dirs, batch_size, num_worker, rank, world_size):
    ds = FaceDataset(data_dirs, p=0.5)
    sampler = ParallelDistributedSampler(ds, world_size, rank, shuffle=True)
    loader = DataLoader(ds,
                        batch_size,
                        shuffle=False,
                        num_workers=num_worker,
                        pin_memory=True,
                        sampler=sampler, 
                        drop_last=False)
    return loader

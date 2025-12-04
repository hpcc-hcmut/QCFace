import argparse
import os
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def str2list(v):
    if ',' in v:
        return v.split(',')
    else:
        return [v]

class ImageData(Dataset):
    def __init__(self, data_dirs):
        super(ImageData, self).__init__()
        self.path_list, self.label_list = self.data_scan(data_dirs)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(
                (0., 0., 0.),
                (1., 1., 1.)
            )
        ])

    def data_scan(self, data_dirs):
        path_list = []
        label_list = []
        idx = 0
        for dir in data_dirs:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trainer for Face Recognition')
    parser.add_argument('--data_dirs', type=str2list,
                        help='<data_root_dir1>,<data_root_dir2>,...')
    parser.add_argument('--device', type=str,
                        help='running device')
    args = parser.parse_args()

    ds = ImageData(args.data_dirs)
    loader = DataLoader(ds, 512,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=False)

    psum    = torch.tensor([0.0, 0.0, 0.0]).to(torch.device(args.device))
    psum_sq = torch.tensor([0.0, 0.0, 0.0]).to(torch.device(args.device))

    for imgs, lbs in tqdm(loader):
        imgs = imgs.to(torch.device(args.device))
        psum += imgs.sum(dim=(0, 2, 3))
        psum_sq += (imgs ** 2).sum(dim=(0, 2, 3))

    count = len(ds) * 112 * 112

    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    print("Statistical features:")
    print("   - mean: {:.4f}".format(total_mean.item()))
    print("   - std:  {:.4f}".format(total_std.item()))

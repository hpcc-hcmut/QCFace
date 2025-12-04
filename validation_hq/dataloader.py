import os
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import path_join

# MEAN = [0.4488, 0.4371, 0.4040]
# STD = [1., 1., 1.]

MEAN = [0.5312, 0.4265, 0.3753]
STD = [0.2873, 0.2555, 0.2496]

class HQFaceDataset(Dataset):
    def __init__(self, dsname, source_dir):
        super(HQFaceDataset, self).__init__()
        self.dsname = dsname
        self.file_list = self.load_list_file(source_dir)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])

    def load_list_file(self, source_dir):
        file_list = []
        for root, _, files in os.walk(source_dir):
            if len(files) < 1:
                continue
            file_list += [path_join(root, file) for file in files]
        return file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), img_path
    
def dataloader_generation(dsname, source_dir, num_workers, batch_size):
    ds = HQFaceDataset(dsname, source_dir)
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader

import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# MEAN = [0.4488, 0.4371, 0.4040]
# STD = [1., 1., 1.]

MEAN = [0.5312, 0.4265, 0.3753]
STD = [0.2873, 0.2555, 0.2496]

class ListDatasetWithIndex(Dataset):
    def __init__(self, img_list):
        super(ListDatasetWithIndex, self).__init__()
        self.img_list = img_list
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx


class ListDataset(Dataset):
    def __init__(self, img_list):
        super(ListDataset, self).__init__()
        self.img_list = img_list
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx


def prepare_imagelist_dataloader(img_list, batch_size, num_workers=0):
    image_dataset = ListDatasetWithIndex(img_list)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return dataloader


def prepare_dataloader(img_list, batch_size, num_workers=0):
    image_dataset = ListDataset(img_list)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader
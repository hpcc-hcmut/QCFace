import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import cv2
import numpy as np
from skimage import transform as trans

# MEAN = [0.4488, 0.4371, 0.4040]
# STD = [1., 1., 1.]

MEAN = [0.5312, 0.4265, 0.3753]
STD = [0.2873, 0.2555, 0.2496]

# From AdaFace src
# LANDMARK_ALIGN = [[30.2946, 51.6963], 
#                   [65.5318, 51.5014], 
#                   [48.0252, 71.7366], 
#                   [33.5493, 92.3655], 
#                   [62.7299, 92.2041]]

# From insightface src
LANDMARK_ALIGN = [[38.2946, 51.6963], 
                  [73.5318, 51.5014], 
                  [56.0252, 71.7366], 
                  [41.5493, 92.3655], 
                  [70.729904, 92.2041]]

class ImageAligner:
    def __init__(self, image_size=(112, 112)):

        self.image_size = image_size
        src = np.array(LANDMARK_ALIGN, dtype=np.float32)
        # if self.image_size[0] == 112:
        #     src[:, 0] += 8.0

        self.src = src

    def align(self, img, landmark):
        # align image with pre calculated landmark

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark

        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(img, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        return img


class ListDatasetWithAligner(Dataset):
    def __init__(self, img_list, landmarks, image_size=(112,112)):
        super(ListDatasetWithAligner, self).__init__()
        self.img_list = img_list
        self.landmarks = landmarks
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112),
            transforms.ToTensor(), 
            transforms.Normalize(MEAN, STD)])
        
        self.aligner = ImageAligner(image_size=image_size)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        landmark = self.landmarks[idx]

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = self.aligner.align(img, landmark)

        if len(img.shape) == 2:
            img = np.stack([img, img, img], -1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)
        return img, idx


def prepare_dataloader(img_list, landmarks, batch_size, num_workers=0, image_size=(112,112)):
    image_dataset = ListDatasetWithAligner(img_list, landmarks, image_size=image_size)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    return dataloader

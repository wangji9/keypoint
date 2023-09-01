import torch
import cv2
import pandas as pd
import numpy as np
import config
import utils
from torch.utils.data import DataLoader,Dataset

from utils import ReadTxt

image_path = r'C:\Users\11711\Desktop\Keypoint-Regression-main\input\FaceKeypoints\training'
train_label_path = r'C:\Users\11711\Desktop\Keypoint-Regression-main\input\FaceKeypoints\training_frames_keypoints.csv'
test_label_path = r'C:\Users\11711\Desktop\Keypoint-Regression-main\input\FaceKeypoints\test_frames_keypoints.csv'
w = 100
h = 100
size = (w,h)

class KeypointDataset(Dataset):

    def __init__(self,samples,size,augment=None):
        self.data =samples
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imread(image_path+'\{}'.format(self.data.iloc[index][0]))
        orig_h, orig_w, channel = img.shape
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,self.size)

        keypoints = self.data.iloc[index][1:]
        keypoints = np.array(keypoints, dtype="uint8")

        '''防止关键点溢出图像'''
        # num_zeros = 0
        # for index, value in enumerate(keypoints):
        #     if value == self.size:
        #         keypoints[index] = self.size - 1
        #     if value == 0:
        #         num_zeros += 1
        #     if value < 0 or value > self.size:
        #         value = np.nan
        # if num_zeros >= 2:
        #     index += 1
        #     return self.__getitem__(index)

        keypoints = keypoints.reshape(68, 2)
        keypoints = keypoints * [self.size[0]/orig_w,self.size[1]/orig_h]


        if self.augment is not None:
            transformed = self.augment(image=img, keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed["keypoints"]


        keypoints_initial = keypoints
        keypoints_augment = keypoints


        image = img / 255.0
        image = np.transpose(image, (2, 0, 1)) #[W,H,C]---> [C,W,H]
        image = torch.tensor(image, dtype=torch.float)

        keypoint = torch.tensor(keypoints, dtype=torch.float)


        if not torch.equal(torch.Tensor([8, 2]), torch.Tensor(list(keypoint.shape))):
            print(f"Keypoints Initial: {keypoints_initial}")
            print(f"Keypoints Augment: {keypoints_augment}")

        # print(keypoints)
        # print(img.shape)

        return {
            'image': image,
            'keypoints': keypoint,
        }


train_data = pd.read_csv(train_label_path)
test_data  = pd.read_csv(test_label_path)

train_data = KeypointDataset(train_data,size)
valid_data = KeypointDataset(test_data,size)

train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
valid_loader = DataLoader(valid_data,batch_size=32,shuffle=False)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")


# if __name__ == '__main__':
#
#     # data = ReadTxt('data.txt')
#     train_data = pd.read_csv(train_label_path)
#     test_data  = pd.read_csv(test_label_path)
#
#
#     dataset = KeypointDataset(data,size,None)
#
#     print(dataset.__getitem__(1))

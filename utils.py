
import matplotlib.pyplot as plt
import numpy as np
import config
import random
import torch
import os

def ReadTxt(txt_file):
    with open(txt_file,'r') as f:
        lines = f.readlines()
    newlines = [line.strip().split(',',1) for line in lines]
    return newlines


def dataset_keypoints_plot(data):
    """
    This function shows the image faces and keypoint plots that the model
    will actually see. This is a good way to validate that our dataset is in
    fact corrent and the faces align wiht the keypoint features. The plot
    will be show just before training starts. Press `q` to quit the plot and
    start training.
    """
    # transform = A.Compose(
    #     [A.HorizontalFlip(p=1)],
    #     keypoint_params=A.KeypointParams(format='xy')
    # )

    plt.figure(figsize=(8, 6))
    for i in range(0, 4):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))

        keypoints = sample['keypoints']
        # print(img.shape)

        # transformed = transform(image=img, keypoints=keypoints)

        # plt.subplot(2, 4, i+1)
        # plt.imshow(transformed['image'])

        # for j in range(len(transformed['keypoints'])):
        #     plt.plot(transformed['keypoints'][j][0], transformed['keypoints'][j][1], 'b.')

        plt.subplot(2, 2, i + 1)
        plt.imshow(img)

        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], '.r')

    plt.show()
    plt.close()
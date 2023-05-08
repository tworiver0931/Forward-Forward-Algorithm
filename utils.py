import torch
from torch.utils.data import Dataset

import numpy as np
import cv2

from tqdm import tqdm
import random


class FFDataset(Dataset):
    def __init__(self, x_pos, x_neg):
        self.x_pos = x_pos.reshape(x_pos.shape[0], -1)
        self.x_neg = x_neg.reshape(x_neg.shape[0], -1)

    def __len__(self):
        return self.x_pos.shape[0]

    def __getitem__(self, idx):
        return self.x_pos[idx], self.x_neg[idx]


class SoftmaxDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_neg(x):
    n = x.shape[0]

    mask = np.random.randint(2, size=(n, 28, 28))
    mask = mask.astype(np.float32)
    print('Create negative datas')
    for i in tqdm(range(n)):
        for _ in range(random.randrange(10, 60)):
            mask[i] = cv2.filter2D(
                mask[i], -1, kernel=np.array([[1/4, 1/2, 1/4]]))
            mask[i] = cv2.filter2D(
                mask[i], -1, kernel=np.array([[1/4], [1/2], [1/4]]))

    mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]

    x_ = x.clone()
    x_neg = x[torch.randperm(n)] * mask + x_[torch.randperm(n)] * (1 - mask)

    return x_neg

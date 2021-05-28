#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 15:33
# @Author  : yangqiang
# @File    : seg_data.py
import shutil
import numpy as np
from torch.utils.data import Dataset
import os
import cv2


class ClsDataset(Dataset):
    def __init__(self, images_dir, file, transform=None):
        self.images_dir = images_dir
        self.mapping_files = self.dict_from_file(file)
        self.transform = transform

    @staticmethod
    def dict_from_file(file):
        mapping = {}
        with open(file, 'r') as f:
            idx = 0
            for line in f:
                items = line.rstrip('\n').split()
                assert len(items) == 2
                mapping[idx] = (items[0], int(items[1]))
                idx += 1
        return mapping

    def __len__(self):
        return len(self.mapping_files)

    def __getitem__(self, idx):
        filename, label = self.mapping_files[idx]

        image = cv2.imread(os.path.join(self.images_dir, filename))
        assert len(image.shape) == 3, f"{filename} is not 3 channels picture"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
        image = image.transpose(2, 0, 1)  # c, h, w

        return image, label


if __name__ == '__main__':
    def test():
        from torch.utils.data import DataLoader
        import albumentations as A

        aug_trans = A.Compose([
            A.Resize(height=260, width=260, p=1),
            A.Normalize(p=1, mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616]),
        ])

        dataset = ClsDataset("../data/images",
                             "../data/val.txt",
                             transform=aug_trans)
        print("len:", len(dataset))
        dataloader = DataLoader(dataset, batch_size=1)
        for image, label in dataloader:
            print(image.shape, label.dtype)
    test()

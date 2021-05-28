#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/4/28 10:45
# @Author  : yangqiang
# @File    : aug.py
import albumentations as A


class CustomAug(object):
    def __init__(self, height, width):
        self.height, self.width = height, width

    def get_train_aug(self):
        train_trans = A.Compose([
            A.Resize(height=self.height, width=self.width, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(p=0.3),
            A.GaussianBlur(p=0.2),
            A.IAASharpen(p=0.4),
            A.CLAHE(p=0.5),
            A.Normalize(p=1, mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616])
        ])
        return train_trans

    def get_val_aug(self):
        val_trans = A.Compose([
            A.Resize(height=self.height, width=self.width, p=1),
            A.Normalize(p=1, mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616])
        ])
        return val_trans

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError("error paras")

        if args[0] == "train":
            return self.get_train_aug()
        elif args[0] == "val":
            return self.get_val_aug()
        else:
            raise NotImplementedError("NotImplemented {} yet".format(args[0]))

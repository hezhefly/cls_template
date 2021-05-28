#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 19:42
# @Author  : yangqiang
# @File    : cls_agent.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from agents.base import BaseAgent
from dataloader import ClsDataset, CustomAug
from loguru import logger
from graphs.efficientnet_pytorch import EfficientNet
import os
from utils.metrics import accuracy, AverageMeter


class ClsAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device(self.cfg.solver.device)  # define device
        self.model = EfficientNet.from_name(self.cfg.model.model_name,
                                            in_channels=self.cfg.model.in_channel,
                                            num_classes=self.cfg.model.num_classes).to(self.device)  # models
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.solver.lr)

        self.current_epoch = 0
        self.best_acc = 0.4

        self.train_trans = CustomAug(height=self.cfg.model.height, width=self.cfg.model.width)('train')
        self.val_trans = CustomAug(height=self.cfg.model.height, width=self.cfg.model.width)('val')
        self.train_data_loader = self.load_dataset('train')
        self.val_data_loader = self.load_dataset('val')

        if self.cfg.model.resume:
            # 预训练模型的参数
            weight_state_dict = torch.load(self.cfg.model.weights, map_location=self.device)
            # model自己的参数
            model_state_dict = self.model.state_dict()
            # 加载预训练模型的参数和model中不冲突的部分
            state_dict_new = {k: v for k, v in weight_state_dict.items() if k in model_state_dict.keys() and
                              self.model.state_dict()[k].shape == v.shape}
            # logger.info("load layers: {}".format(state_dict_new.keys()))
            model_state_dict.update(state_dict_new)
            # 更新模型参数
            self.model.load_state_dict(model_state_dict)

    def load_dataset(self, mode):
        if mode == "train":
            train_dataset = ClsDataset(images_dir=self.cfg.datasets.images_dir,
                                       file=self.cfg.datasets.train_file,
                                       transform=self.train_trans)
            train_data_loader = DataLoader(train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True)
            logger.info("%d train data loaded,batch size: %d" % (len(train_data_loader.dataset),
                                                                 self.cfg.train.batch_size))
            return train_data_loader
        elif mode == "val":
            val_dataset = ClsDataset(images_dir=self.cfg.datasets.images_dir,
                                     file=self.cfg.datasets.val_file,
                                     transform=self.val_trans)
            val_data_loader = DataLoader(val_dataset, batch_size=self.cfg.test.batch_size)
            logger.info("%d val data loaded,batch size: %d" % (len(val_data_loader.dataset),
                                                               self.cfg.test.batch_size))
            return val_data_loader

    @logger.catch
    def train(self):
        for epoch in range(self.cfg.train.max_epoch):
            self.model.train()
            for batch_idx, (images, labels) in enumerate(self.train_data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                logger.info('Train Epoch: [{}/{}] [{}/{}], current batch Loss: {:.6f} lr: {}'.format(
                    self.current_epoch,
                    self.cfg.train.max_epoch,
                    batch_idx * self.cfg.train.batch_size + len(images),
                    len(self.train_data_loader.dataset),
                    loss.item(),
                    self.optimizer.state_dict()['param_groups'][0]['lr']))

            self.current_epoch += 1

            if self.current_epoch % self.cfg.save.val_per_epoch == 0:
                acc = self.validate()
                acc_best = acc > self.best_acc
                if acc_best:  # 平均iou变大
                    self.best_acc = acc
                    save_path = os.path.join(self.cfg.save.weight_dir,
                                             f"{self.cfg.model.name}_{self.current_epoch}_{round(acc, 3)}.pt")
                    torch.save(self.model.state_dict(), save_path)

    def load_model(self, saved_model):
        self.model.load_state_dict(torch.load(saved_model, map_location=self.device))
        self.model.eval()

    def validate(self):
        self.model.eval()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        with torch.no_grad():
            for images, labels in self.val_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss(outputs, labels)

                acc1 = accuracy(outputs, labels, topk=(1,))[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

            logger.info(f"val loss:{losses.avg}, acc: {top1.avg}")
            return top1.avg

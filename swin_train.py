#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/5/28 13:02
# @Author  : yangqiang
# @File    : swin_train.py
from agents.swin_agent import ClsAgent
from dotmap import DotMap
import yaml


def main():
    cfg = DotMap(yaml.load(open("configs/swin.yaml"), Loader=yaml.FullLoader))
    agent = ClsAgent(cfg)
    agent.train()


if __name__ == '__main__':
    main()

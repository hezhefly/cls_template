#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/5/27 19:47
# @Author  : yangqiang
# @File    : trian.py
from agents import ClsAgent
from dotmap import DotMap
import yaml


def main():
    cfg = DotMap(yaml.load(open("configs/efficientnet.yaml"), Loader=yaml.FullLoader))
    agent = ClsAgent(cfg)
    agent.train()


if __name__ == '__main__':
    main()

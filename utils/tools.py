#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/27 19:30
# @Author  : yangqiang
# @File    : tools.py
import time
import logging


def recursive_update(default, custom):
    """
    example:
        default = {'a': {'c': {'d': 2, "e": 3}}, 'b': 4}
        custom = {'a': {'c': {"f": 9}, 'd': {1, 2}}}
        recursive_update(default, custom)  # {'a': {'c': {'d': 2, 'e': 3, 'f': 9}, 'd': {1, 2}}, 'b': 4} ✅
        # default.update(custom)  # {'a': {'c': {'f': 9}, 'd': {1, 2}}, 'b': 4} ❌
    """
    if not isinstance(default, dict) or not isinstance(custom, dict):
        raise TypeError('Params of recursive_update should be dicts')

    for key in custom:
        if isinstance(custom[key], dict) and isinstance(
                default.get(key), dict):
            default[key] = recursive_update(default[key], custom[key])
        else:
            default[key] = custom[key]

    return default


def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
                                        (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    call(["nvcc", "--version"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))

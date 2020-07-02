#!/usr/bin/env python
# coding=utf-8
'''
@Author: xieydd
@since: 2020-06-11 18:37:43
@lastTime: 2020-06-12 19:06:58
@LastAuthor: Do not edit
@message: 
'''
cfg_slim = {
    'name': 'slim',
    # 'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'min_sizes': [[8, 10, 16, 24], [30, 42, 56], [64, 80, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    # 'batch_size': 320,
    # 'ngpu': 1,
    # 'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300
}

import os
import time
import numpy as np
import torch
import sys
import pickle
import json
import shutil
import math


def entropy(p):
    return np.nansum(-p * np.log(p))


def to_numpy(var, toint=False):
    #  Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    if isinstance(var, torch.Tensor):
        var = var.squeeze().detach().cpu().numpy()
    if toint:
        var = var.astype('uint8')
    return var


# pickle io
def dump_pickle(data, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
        print('write data to', out_path)


def load_pickle(in_path):
    with open(in_path, 'rb') as f:
        data = pickle.load(f)  # list
        return data


# json io
def dump_json(adict, out_path):
    with open(out_path, 'w', encoding='UTF-8') as json_file:
        # 设置缩进，格式化多行保存; ascii False 保存中文
        json_str = json.dumps(adict, indent=2, ensure_ascii=False)
        json_file.write(json_str)


def load_json(in_path):
    with open(in_path, 'rb') as f:
        adict = json.load(f)
        return adict


# io: txt <-> list
def write_list_to_txt(a_list, txt_path):
    with open(txt_path, 'w') as f:
        for p in a_list:
            f.write(p + '\n')


def read_txt_as_list(f):
    with open(f, 'r') as f:
        return [p.replace('\n', '') for p in f.readlines()]


def approx_print(arr, decimals=2):
    arr = np.around(arr, decimals)
    print(','.join(map(str, arr)))


def recover_color_img(img):
    """
    cvt tensor image to RGB [note: not BGR]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.transpose(img, axes=[1, 2, 0])  # h,w,c
    img = img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)  # 直接通道相成?
    img = (img * 255).astype('uint8')
    return img


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time


class Logger:
    """logger"""
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='UTF-8')  # 打开时自动清空文件

    def write(self, msg):
        self.terminal.write(msg)  # 命令行打印
        self.log.write(msg)

    def flush(self):  # 必有，不然 AttributeError: 'Logger' object has no attribute 'flush'
        pass

    def close(self):
        self.log.close()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

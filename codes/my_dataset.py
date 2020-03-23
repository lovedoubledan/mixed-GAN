'''
loading dataset
@author zhandandan
'''
from PIL import Image
import os
import glob
import random
import shutil
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import torchvision
import matplotlib.pyplot as plt
import numpy as np

my_list = {"Boat Shoes":0, "Boots":1,"Loafers":2, "Oxfords":3, "Sneakers and Athletic Shoes":4}
'''
my_list  = {"Boat Shoes":0, "Boots":1, "Clogs and Mules":2, "Filp Flops":3, "Firstwalker":4, "Flats":5, 
           "Heels":6, "Loafers":7, "Oxfords":8, "Sandals":9, "Slippers":10, "Sneakers and Athletic Shoes":11}
'''
def get_list(data_path): 
    img_list = []
    label = 0
    img_label_list = my_list
    for root, s_dirs, _ in os.walk(data_path, topdown=True):   # 获取data文件下各文件夹名称      
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)                # 获取各类的文件夹 绝对路径
            all_img_path = os.listdir(i_dir)                   # 获取类别文件夹下所有png图片的路径
            for i in range(len(all_img_path)):
                if not all_img_path[i].endswith('jpg'):        # 若不是png文件，跳过
                    continue
                for key in img_label_list:
                    if key in root:
                        label = img_label_list[key]
                        break
                
                img_path = os.path.join(i_dir, all_img_path[i])
                img_list.append([img_path, label])
    
    return img_list

class MyDataset(Dataset):
    def __init__(self, data_path, transform = None, target_transform = None):
        imgs = []
        imgs = get_list(data_path)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path) 
        if self.transform is not None:
            img = self.transform(img) 
        else:
            img = transforms.ToTensor()(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
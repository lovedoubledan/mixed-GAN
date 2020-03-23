'''
finding the single channel image that causes bug
@author : wujiahao
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

data_path = os.path.join("B:/", "deeplearningbigwork", "my_code","data")
imgs = get_list(data_path)
for im_path, label in imgs:
    img = Image.open(im_path)
    if len(img.split()) != 3:
        print(im_path, label)
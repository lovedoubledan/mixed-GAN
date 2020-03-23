'''
script used to visulize dataset on different classes
@author: zhandandan
'''
from PIL import Image
import os
import glob
import random
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import my_dataset
import numpy as np
            
data_dir = os.path.join("F:/", "myfile", "deeplearning","data (2)", "data")
img_list = []
img_list = my_dataset.get_list(data_path=data_dir)
my_list = []
for i in range(12): 
    for img_dir in img_list:
        path, label = img_dir
        if label == i:
            img = Image.open(path)
            img = transforms.ToTensor()(img)
            my_list.append(img)
            break
            
my_tensor = my_list[0].unsqueeze(0) 
for i in range(1,12):
    my_tensor = torch.cat((my_tensor, my_list[i].unsqueeze(0)), 0)

img = torchvision.utils.make_grid(my_tensor, nrow=4)
img = np.transpose(img, (1, 2, 0))
plt.imshow(img)
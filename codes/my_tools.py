'''
save image during training process for observation
first contributed by zhandandan, debug jobs, some slight modifies and GPU parallel codes from wujiahao 
@author: zhandandan, wujiahao
'''
import numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import config
import os

opt = config.my_config()
latent_dim = opt.latent_dim
n_classes = opt.n_classes
img_size = opt.img_size
channels = opt.channels
code_dim = opt.code_dim

def sample_image(n_row, batches_done, generator, cuda, epoch):
    if not os.path.exists("images/test/"):
        os.makedirs("images/test/", exist_ok=True)

    device_for_data = torch.device('cuda:0' if cuda else 'cpu')
    device_for_model = torch.device('cuda' if cuda else 'cpu')
    
    generator.eval()
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    for time in range(4):
        z = Variable(FloatTensor(np.random.normal(0, 1, (8 ** 2, latent_dim)))).to(device_for_data) 
        for label in range(n_classes):
            labels = Variable(LongTensor(np.repeat([label], 8 ** 2))).to(device_for_data)         
            c_varied1 = np.linspace(-1, 1, 8)[:, np.newaxis]
            c_varied2 = np.linspace(-1, 1, 8, endpoint=False)[:, np.newaxis]
            code = Variable(FloatTensor([[i[0], j[0]] for i in c_varied1 for j in c_varied2])).to(device_for_data)
            img = generator(z, labels, code)
            path = "images/test/epoch_%d"%epoch + "/label_%d/"%label 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            save_image(img, path + "z_%d.png"%time, nrow=8, normalize=True)
    generator.train()
    
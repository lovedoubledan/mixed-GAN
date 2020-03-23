'''
main function here, training process here
codes implementing dataloader, transform, parameter save are from zhandandan, others from wujiahao
main training process modified from https://gitee.com/xiaonaw/PyTorch-GAN/blob/master/implementations , almost all changed
author: wujiahao, zhandandan
'''
import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.nn import DataParallel

import my_model
import my_tools
import my_dataset
import config



opt = config.my_config()
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
data_dir = opt.data_dir
isSaveArgument = opt.isSaveArgument
retrain = opt.retrain
G_para_file = opt.G_para_file
D_para_file = opt.D_para_file
iterNumToSave = opt.iterNumToSave
lambda_gp = opt.lambda_gp
lambda_con = opt.lambda_con
lambda_mismatch = opt.lambda_mismatch
G_savePath = opt.G_savePath
D_savePath = opt.D_savePath

# 保存参数到哪个路径 
os.makedirs(G_savePath, exist_ok=True)
os.makedirs(D_savePath, exist_ok=True)
# -----------------------------------------------------------------------------------

# Loss functions
continuous_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = my_model.Generator()
discriminator = my_model.Discriminator()

device_for_data = torch.device('cuda:0' if cuda else 'cpu')
device_for_model = torch.device('cuda' if cuda else 'cpu')

if cuda:
    generator = DataParallel(generator)
    generator.to(device_for_model)
    discriminator = DataParallel(discriminator)
    discriminator.to(device_for_model)

# Initialize weights
generator.apply(my_model.weights_init_normal)
discriminator.apply(my_model.weights_init_normal)



transform_train = transforms.Compose([transforms.Resize((256,256)),             
                                     transforms.ToTensor(),])
dataset = my_dataset.MyDataset(data_path= data_dir, transform=transform_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr * 4, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------
if retrain == False:
    generator.load_state_dict(torch.load(G_para_file))
    discriminator.load_state_dict(torch.load(D_para_file))

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor)).to(device_for_data)
        labels = Variable(labels.type(LongTensor)).to(device_for_data)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))).to(device_for_data)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim)))).to(device_for_data)

        # Generate a batch of images
        gen_imgs = generator(z, labels, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs, labels)
        g_loss = -torch.mean(validity)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images and real label
        validity_real, _ = discriminator(real_imgs, labels)

        # Loss for fake images 
        validity_fake, _ = discriminator(gen_imgs.detach(), labels)

        # gradient penalty loss
        gradient_penalty = my_model.compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, labels, cuda)

        # Total discriminator loss
        d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty 
        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size))).to(device_for_data)


        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))).to(device_for_data)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim)))).to(device_for_data)

        gen_imgs = generator(z, sampled_labels, code_input)
        _, pred_code = discriminator(gen_imgs, sampled_labels)

        info_loss = lambda_con * continuous_loss(pred_code, code_input)
        info_loss.backward()
        optimizer_info.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.cpu().item(), g_loss.cpu().item(), info_loss.cpu().item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            my_tools.sample_image(n_row=opt.n_classes, batches_done=batches_done, generator=generator, cuda=cuda, epoch=epoch)
        
    # 保存参数
    if isSaveArgument == True and epoch % iterNumToSave == 0:
        save_times = epoch / iterNumToSave 
        G_save_path = os.path.join(G_savePath, "G_para%d.pth" % save_times)
        D_save_path = os.path.join(D_savePath, "D_para%d.pth" % save_times)
        torch.save(generator.state_dict(), G_save_path)
        torch.save(discriminator.state_dict(), D_save_path)

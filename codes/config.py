'''
All the configs and parameters using in training and testing are defined and modified here
author : wujiahao
'''
import os
class my_config():
    def __init__(self):
        self.n_epochs = 200
        self.batch_size = 32
        self.lr = 6e-5
        self.b1 = 0.5   # adam: decay of first order momentum of gradient
        self.b2 = 0.999 # adam: decay of first order momentum of gradient
        self.n_cpu = 8 # number of cpu threads to use during batch generation
        self.latent_dim = 100 # dimensionality of the latent space
        self.n_classes = 5  # number of classes for dataset
        self.img_size = 256  # size of each image dimension
        self.channels = 3  # number of image channels
        self.sample_interval = 400  # interval between image sampling
        self.code_dim = 2 # latent code
        self.clip_value = 0.01  # lower and upper clip value for disc. weights
        self.data_dir = "/GPUFS/nsccgz_ywang_1/wujiahao/gan/data2"
        self.isSaveArgument = True  # 是否保存参数
        self.retrain = True # 是否重新开始训练 
        self.G_para_file = "" 
        self.D_para_file = ""
        self.iterNumToSave = 10 # 每过几个迭代保存一次参数
        self.lambda_gp = 10 # Loss weight 
        self.lambda_con = 10
        self.G_savePath = "my_argument/"
        self.D_savePath = "my_argument/"



       

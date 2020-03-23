# mixed-GAN

### It was my groups' final assignment of course deep learning, which mixed several exited GANs to create a GAN of our own. It was used on a shoes dataset to generate shoes images


### run:
- down load dataset 
- change config in config.py
- run 
  > python my_code.py

### requirement:
- python >= 3.6
- pytorch >= 1.0

### dataset download link
- baiduyun link: [https://pan.baidu.com/s/1oxmfGmgAwPoxkY1R5Yk6_g](https://pan.baidu.com/s/1oxmfGmgAwPoxkY1R5Yk6_g) 
password: xeww
- 12 classes dataset in [data1], 5 classes dataset in [data2]
- network parameters

### NOTE:
#### if you want to run on your own dataset, you should change 
- my_dataset.py
- imsize and num_classes in config.py 
#### if you want to evaluate on your self, my_tools.py can be referenced, it hasn't been debuged

--- 
## 中文：
### 这是我们组在深度学习课程的期末大作业， 我们融合了几个现有的GAN模型，并把它用在一个鞋子的数据集上面，它可以生成各自各样的鞋子
  
### 运行：
配置文件在 config.py里，数据集下载后修改config.py里面的数据集路径和batchsize之类的配置
然后python my_code.py 就可以了
  
### 环境要求：
python 3.6以上
pytorch 1.0以上，若想在GPU上运行需要配置好pytorch的GPU版本
  
### 参数、12分类数据集【data1】、5分类数据集【data2】 下载
链接: https://pan.baidu.com/s/1oxmfGmgAwPoxkY1R5Yk6_g 
提取码: xeww
  
### NOTE:
若想在自己的数据集上运行，只需要修改my_dataset.py 即可， 对应imsize和分类数在config.py中修改
evaluate 可以参照my_tool.py 

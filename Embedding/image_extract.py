import random
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import timm
# from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model
from timm.models.vision_transformer import vit_base_patch16_224 as create_model

# resnet18 = models.resnet18()
# vit = create_model(pretrained=True)
vit = create_model(pretrained=True)
# vit.load_state_dict(torch.load('pre_tune_vit1.pth'))

handle = transforms.Compose([transforms.RandomResizedCrop(224),  # 将PIL图像裁剪成任意大小和纵横比
                             transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
                             transforms.RandomVerticalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])


def Image_Extractor(path, handle=handle):
    image = Image.open(path).convert("RGB")
    image = handle(image)
    image = torch.unsqueeze(image, 0)
    output = vit(image)
    output_feat = vit.forward_features(image)
    return output.cpu().detach(), output_feat.cpu().detach()


def demo(root):
    path = root + '/data/post_video/007大破天幕杀机/image/海报1.jpg'  # 540*774图片
    image = Image.open(path).convert('RGB')

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = img_transforms(image)  # H*W*C
    print(image.shape)


if __name__ == '__main__':
    root = 'G:/Pytorch/Movies_Predict/Office_Prediction'
    # root = '/home/20215227060qmf/Movies_Work/Office_Prediction'
    # demo(root)  # 试验
    path = root + '/data/post_video/乌尔善/刀见笑/image/剧照1.jpg'
    output, output_feat = Image_Extractor(path)
    print(output.shape, output_feat.shape)
    print(output_feat)
    '''
    paths = "['./data/post_video/乌尔善/刀见笑/image/剧照1.jpg', './data/post_video/乌尔善/刀见笑/image/剧照2.jpg', './data/post_video/乌尔善/刀见笑/image/剧照3.jpg', './data/post_video/乌尔善/刀见笑/image/剧照4.jpg', './data/post_video/乌尔善/刀见笑/image/剧照5.jpg', './data/post_video/乌尔善/刀见笑/image/海报1.jpg', './data/post_video/乌尔善/刀见笑/image/海报2.jpg', './data/post_video/乌尔善/刀见笑/image/海报3.jpg', './data/post_video/乌尔善/刀见笑/image/海报4.jpg', './data/post_video/乌尔善/刀见笑/image/海报5.jpg']"

    output, output_feat, output_mean = Image_Extractor(paths)
    print(output.shape, output_feat.shape, output_mean.shape)
    print(output_mean)
    '''

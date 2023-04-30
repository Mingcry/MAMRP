# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import os
from random import *
import subprocess
from torch.autograd import Variable
from Embedding.Pre_train_model.timesformer.models.transforms import *
from Embedding.Pre_train_model.timesformer.models.vit import TimeSformer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_input(image_path, num_frame=16, flag=False):
    prefix = '{:05d}.jpg'
    feat_path = image_path
    images = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_params = {
        "side_size": 256,
        "crop_size": 224,
        "num_segments": num_frame,
        "sampling_rate": 5
    }
    transform_val = torchvision.transforms.Compose([
        GroupScale(int(transform_params["side_size"])),
        GroupCenterCrop(transform_params["crop_size"]),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(mean, std),
    ])
    frame_list = os.listdir(feat_path)
    average_duration = len(frame_list) // transform_params["num_segments"]
    offsets = np.array(
        [int(average_duration / 2.0 + average_duration * x) for x in range(transform_params["num_segments"])])
    offsets = offsets + 1
    if flag:
        print('采样坐标：', offsets)
    for i, seg_ind in enumerate(offsets):
        p = int(seg_ind)
        seg_imgs = Image.open(os.path.join(feat_path, prefix.format(p))).convert('RGB')
        if i == 0 and flag:
            print('*******每帧大小：', seg_imgs.size)
        images.append(seg_imgs)
    video_data = transform_val(images)
    if flag:
        print('帧数量：', len(images))
        print('视频数据初始shape：', video_data.shape)
    video_data = video_data.view((-1, transform_params["num_segments"]) + video_data.size()[1:])
    if flag:
        print("视频数据最终shape：", video_data.shape)
    out = Variable(video_data)

    return out.unsqueeze(0).to(device)


def handle(root, path, num_frame, flag=False):
    movie_name = path.split('/')[-3]
    out_image_dir = root + '/Embedding/video_key_frame/' + movie_name
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)

    # =================视频抽帧======================
    video_name = path.split('/')[-1].split('.')[0]
    out_image_path = os.path.join(out_image_dir, video_name)
    if not os.path.exists(out_image_path):
        os.makedirs(out_image_path)
    if len(os.listdir(out_image_path)) == 0:
        ffmepg = 'E:/download/ffmpeg-5.1.2-essentials_build/bin/ffmpeg.exe '
        # ffmepg = '/home/20215227060qmf/ffmepg_file/ffmpeg-git-20220910-amd64-static/ffmpeg '
        cmd = ffmepg + "-i \"{}\" -vf select='eq(pict_type\,I)' -vsync 2 -s 224*224 -f image2 \"{}/%05d.jpg\"".format(path,
                                                                                                                     out_image_path)
        if flag:
            print('cmd=', cmd)
        r = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert r == 0, "抽帧失败！"
        if flag:
            print('抽帧完成!')

    # =================提取特征======================
    feat = get_input(out_image_path, num_frame=num_frame, flag=flag)
    if flag:
        print('feat:', feat.shape)
    return feat


def extract(root, model_input):
    # =================模型建立======================
    model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
                        pretrained_model=root+'/Embedding/Pre_train_model/timesformer/TimeSformer_divST_8x32_224_K600.pyth')

    model = model.eval().to(device)
    classifer, feature = model(model_input)
    feature = feature.cpu().detach()
    return feature


def Video_Extractor(root, path, num_frame=16, flag=False):
    video_path = path
    if flag:
        print('=================')
        print(video_path)
    feat = handle(root, video_path, num_frame, flag=flag)

    if flag:
        print('===================================')
        print('num_video:', len(feat))
        print('model_input:', feat.shape)
    output = extract(root, feat)
    if flag:
        print('model_output:', output.shape)
    return output


if __name__ == '__main__':
    num_frame = 16  # 帧数目
    flag = True  # 是否输出中间print

    # 输入形式
    root = 'G:/Pytorch/Movies_Predict/Office_Prediction'
    # root = '/home/20215227060qmf/Movies_Work/Office_Prediction'
    path = root + '/data/post_video/007大破天幕杀机/video/视频1.mp4'

    output = Video_Extractor(root, path, num_frame=num_frame, flag=flag)
    # embedding_path = root + '/Embedding/video_embeddings/' + path.split('/')[-3] + '.pt'
    # a = torch.load(embedding_path)
    # print(a)
    # print(a.shape)
    print(output.shape)

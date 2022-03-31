"""
本py文件是运用在项目运行中的模型运输部分，功能是对一整个文件夹中的图片进行模型检测，如果比判断为deepfake的图片的概率大于50%，则判断为deepfake视频
"""


import glob
import os
import torch
from PIL import Image
from scan import *
from torchvision import transforms
import cv2


# 文件夹的路径、总体的图片计数、发现deepfake的计数
path_name = r"D:\DeepFakeProject_in_D\deepfake_project\eliminate_project\secruity-eye\res\006_002"
total_count = 0
deepfake_count = 0

listdir = os.listdir(path_name)

# 模型的配置文件  地址是同级目录下的json文件 同时json文件里面的pretrain_path需要修改
with open(r"D:\DeepFakeProject_in_D\deepfake_project\our_code\deepwake\config.json") as f:
    cfg = json.loads(f.read())

arch = cfg['arch']
margin = cfg['margin']
face_size = (cfg['size'], cfg['size'])

print(f'margin: {margin}, size: {face_size}, arch: {arch}')

# 模型调用
model_list = []
model = EffNet(arch).to("cuda:0")
checkpoint = torch.load(cfg['pretain_model_path'], map_location=lambda storage, loc: storage.cuda(0))
# checkpoint = torch.load(cfg['pretain_model_path'], map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)
del checkpoint

model_list.append(model)
deepware = Ensemble(model_list).eval().to("cuda:0")
# deepware = Ensemble(model_list).eval()

# 开始使用
for file in listdir:
    file_name = path_name+"\\"+file

    tf = transforms.Compose([
        lambda x: Image.open(x).convert("RGB"),  # string path => image data
        transforms.Resize(face_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = tf(file_name)
    img = img.unsqueeze(0)

    output = deepware(img.to("cuda:0"))
    # output = deepware(img)
    print(output)
    if output.item() > 0 :
        # print("图片为deepfake")
        deepfake_count+=1
    # else:
        # print("图片为真")

    total_count+=1

if deepfake_count / total_count < 0.2:
    print("这个视频是真实的视频")
else:
    print("这个视频是deepfake视频")


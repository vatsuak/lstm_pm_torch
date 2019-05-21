# predictor
from data.handpose_data2 import UCIHandPoseDataset
from model.lstm_pm import LSTM_PM
from src.utils import *
# from __future__ import print_function
import argparse
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.data import DataLoader
from printer import pred_images

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# add parameter


learning_rate = 8e-6
batch_size = 1
# save_dir = ckpt
cuda = 1

# hyper parameter
temporal = 5
test_data_dir = './dataset/train_data'
test_label_dir = './dataset/train_label'
# model_epo = [50]

# load data
test_data = UCIHandPoseDataset(data_dir=test_data_dir, label_dir=test_label_dir, temporal=temporal, train=False)
print('Test  dataset total number of images sequence is ----' + str(len(test_data)))
test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False)


def load_model():
    # build model
    net = LSTM_PM(T=temporal)
    net = net.cuda()
    # save_path = os.path.join()
    net.load_state_dict(torch.load('./ckpt2/ucihand_lstm_pm70.pth'))
    return net

# **************************************** test all images ****************************************


print('********* test data *********')

net = load_model()
net.eval()
outp = []
for step, (images, label_map, center_map, imgs) in enumerate(test_dataset):
    
    images = Variable(images.cuda())          # 4D Tensor
    # Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
    label_map = Variable(label_map.cuda())  # 5D Tensor
    # Batch_size  *  Temporal        * joint *   45 * 45
    center_map = Variable(center_map.cuda())  # 4D Tensor
    # Batch_size  *  1          * width(368) * height(368)
    predict_heatmaps = net(images, center_map)  # get a list size: temporal * 4D Tensor
    predict_heatmaps =  predict_heatmaps[1:]
    out = pred_images(predict_heatmaps, step, temporal=temporal)
    pd.DataFrame(out).to_csv('./values/'+str(step)+'.csv', header=None, index=None)
#     outp.append(out)
# print(outp[1].max)


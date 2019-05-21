# https://github.com/HowieMa/lstm_pm_pytorch.git
import argparse
from model.lstm_pm import LSTM_PM
from data.handpose_data2 import UCIHandPoseDataset
from src.utils import *

import torch
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

device_ids = [0]

# hyper parameter
temporal = 5
train_data_dir = './dataset/train_full_data/'
train_label_dir = './dataset/train_label/'

# add parameter
# parser = argparse.ArgumentParser(description='Pytorch LSTM_PM with Penn_Action')
# parser.add_argument('--learning_rate', type=float, default=8e-6, help='learning rate')
# parser.add_argument('--batch_size', default=3, type=int, help='batch size for training')
# parser.add_argument('--epochs', default=50, type=int, help='number of epochs for training')
# parser.add_argument('--begin_epoch', default=0, type=int, help='how many epochs the model has been trained')
# parser.add_argument('--save_dir', default='./ckpt/', type=str, help='directory of checkpoint')
# parser.add_argument('--cuda', default=1, type=int, help='if you use GPU, set cuda = 1,else set cuda = 0')
# parser.add_argument('--temporal', default=4, type=int, help='how many temporals you want ')
# args = parser.parse_args()

learning_rate = 8e-6
batch_size = 2
epochs = 100
begin_epoch = 50
save_dir = './ckpt2/'
cuda = 1

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

transform = transforms.Compose([transforms.ToTensor()])

# Build dataset
train_data = UCIHandPoseDataset(data_dir=train_data_dir, label_dir=train_label_dir, temporal=temporal, train=True)
print(len(train_data))
print('Train dataset total number of images sequence is ----' + str(len(train_data)))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Build model
net = LSTM_PM(T=temporal)
net = net.cuda()
net.load_state_dict(torch.load('./ckpt2/ucihand_lstm_pm50.pth'))
def train():
    # initialize optimizer
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = StepLR(optimizer, step_size=40000, gamma=0.333)

    criterion = nn.MSELoss(size_average=True)                       # loss function MSE average

    net.train()
    for epoch in range(begin_epoch, epochs + 1):

        print('epoch....................................' + str(epoch))
        for step, (images, label_map, center_map, imgs) in enumerate(train_dataset):

            images = Variable(images.cuda())               # 4D Tensor
            # Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
            label_map = Variable(label_map.cuda())      # 5D Tensor
            # Batch_size  *  Temporal        * (joints+1) *   45 * 45
            center_map = Variable(center_map.cuda())   # 4D Tensor
            # Batch_size  *  1          * width(368) * height(368)

            optimizer.zero_grad()
            predict_heatmaps = net(images, center_map)  # get a list size: (temporal + 1 ) * 4D Tensor

            # ******************** calculate and save loss of each joints ********************
            total_loss = save_loss(predict_heatmaps, label_map, epoch, step, criterion, train=True, temporal=temporal)
            # if step % 10 == 0:
            #     print('--step .....' + str(step))
            #     print('--loss ' + str(float(total_loss)))

            # ******************** save training heat maps per 100 steps ********************
            # if step % 100 == 0:
            #     save_images(label_map, predict_heatmaps, step, epoch, imgs, train=True, temporal=temporal)

            # backward
            total_loss.backward()
            optimizer.step()
            # scheduler.step()

        #  ************************* save model per 10 epochs  *************************
        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'ucihand_lstm_pm{:d}.pth'.format(epoch)))
        print('--step .....' + str(step))
        print('--loss ' + str(float(total_loss)))

    print('train done!')


if __name__ == '__main__':
    train()









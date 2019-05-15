import json
import numpy as np
import os
import scipy.misc


def loss_history_init(temporal=5):
    loss_history = {}
    for t in range(temporal):
        loss_history['temporal'+str(t)] = []
    loss_history['total'] = 0.0
    return loss_history


def save_loss(predict_heatmaps, label_map, epoch, step, criterion, train, temporal=5, save_dir='ckpt/'):
    loss_save = loss_history_init(temporal=temporal)

    predict = predict_heatmaps[0]
    target = label_map[:, 0, :, :, :]
    initial_loss = criterion(predict, target)  # loss initial
    total_loss = initial_loss

    for t in range(temporal):
        predict = predict_heatmaps[t + 1]
        target = label_map[:, t, :, :, :]
        tmp_loss = criterion(predict, target)  # loss in each stage
        total_loss += tmp_loss
        loss_save['temporal' + str(t)] = float('%.8f' % tmp_loss)

    total_loss = total_loss
    loss_save['total'] = float(total_loss)

    # save loss to file
    if train is True:
        if not os.path.exists(save_dir + 'loss_epoch' + str(epoch)):
            os.mkdir(save_dir + 'loss_epoch' + str(epoch))
        json.dump(loss_save, open(save_dir + 'loss_epoch' + str(epoch) + '/s' + str(step).zfill(4) + '.json', 'w', encoding="utf8"))

    else:
        if not os.path.exists(save_dir + 'loss_test/'):
            os.mkdir(save_dir + 'loss_test/')
        json.dump(loss_save, open(save_dir + 'loss_test/' + str(step).zfill(4) + '.json', 'w', encoding="utf8"))

    return total_loss


def save_images(label_map, predict_heatmaps, step, epoch, imgs, train, pck=1, temporal=5, save_dir='ckpt/'):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal * joints *   45 * 45
    :param step:
    :param temporal:
    :param epoch:
    :param train:
    :param imgs: list [(), (), ()] temporal * batch_size
    :return:
    """

    for b in range(label_map.shape[0]):                     # for each batch (person)
        output = np.ones((50 * 2, 50 * temporal))           # cd .. temporal save a single image
        seq = imgs[0][b].split('/')[-2]                     # sequence name 001L0
        img = ""
        for t in range(temporal):                           # for each temporal
            im = imgs[t][b].split('/')[-1][1:5]             # image name 0005
            img += '_' + im
            pre = np.zeros((45, 45))  #
            gth = np.zeros((45, 45))
            for i in range(21):                             # for each joint
                pre += np.asarray((predict_heatmaps[t][b, i, :, :].data).cpu())  # 2D
                gth += np.asarray((label_map[b, t, i, :, :].data).cpu())        # 2D

            output[0:45,  50 * t: 50 * t + 45] = gth
            output[50:95, 50 * t: 50 * t + 45] = pre

        if train is True:
            if not os.path.exists(save_dir + 'epoch'+str(epoch)):
                os.mkdir(save_dir + 'epoch'+str(epoch))
            scipy.misc.imsave(save_dir + 'epoch'+str(epoch) + '/s'+str(step) + '_b' + str(b) + '_' + seq + img + '.jpg', output)
        else:

            if not os.path.exists(save_dir + 'test'):
                os.mkdir(save_dir + 'test')
            scipy.misc.imsave(save_dir + 'test' + '/s' + str(step) + '_b' + str(b) + '_'
                              + seq + img + '_' + str(round(pck, 4)) + '.jpg', output)


def lstm_pm_evaluation(label_map, predict_heatmaps, sigma=0.04, temporal=5):
    pck_eval = []
    empty = np.zeros((21, 45, 45))                                      # 3D numpy 21 * 45 * 45
    for b in range(label_map.shape[0]):        # for each batch (person)
        for t in range(temporal):           # for each temporal
            target = np.asarray((label_map[b, t, :, :, :].data).cpu())          # 3D numpy 21 * 45 * 45
            predict = np.asarray((predict_heatmaps[t][b, :, :, :].data).cpu())  # 3D numpy 21 * 45 * 45
            if not np.equal(empty, target).all():
                pck_eval.append(PCK(predict, target, sigma=sigma))

    return sum(pck_eval) / float(len(pck_eval))  #


def PCK(predict, target, label_size=45, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         3D numpy       21 * 45 * 45
    :param target:          3D numpy       21 * 45 * 45
    :param label_size:
    :param sigma:
    :return: 0/21, 1/21, ...
    """
    pck = 0
    for i in range(predict.shape[0]):
        pre_x, pre_y = np.where(predict[i, :, :] == np.max(predict[i, :, :]))
        tar_x, tar_y = np.where(target[i, :, :] == np.max(target[i, :, :]))

        dis = np.sqrt((pre_x[0] - tar_x[0])**2 + (pre_y[0] - tar_y[0])**2)
        if dis < sigma * label_size:
            pck += 1
    return pck / float(predict.shape[0])


def draw_loss(epoch):
    all_losses = os.listdir('ckpt/loss_epoch'+str(epoch))
    losses = []

    for loss_j in all_losses:
        loss = json.load('ckpt/loss_epoch'+str(epoch) + '/' +loss_j)
        a = loss['total']
        losses.append(a)


def Tests_save_label_imgs(label_map, predict_heatmaps, step, imgs, temporal=13, save_dir='ckpt/'):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal * joints *   45 * 45
    :param step:
    :param temporal:
    :param epoch:
    :param train:
    :param imgs: list [(), (), ()] temporal * batch_size
    :return:
    """

    for b in range(label_map.shape[0]):  # for each batch (person)
        output = np.ones((50 * 2, 50 * temporal))  # cd .. temporal save a single image
        seq = imgs[0][b].split('/')[-2]  # sequence name 001L0
        img = ""  # all image name in the same seq
        label_dict = {}  # all image label in the same seq
        pck_dict = {}
        for t in range(temporal):  # for each temporal
            labels_list = []  # 21 points label for one image [[], [], [], .. ,[]]

            im = imgs[t][b].split('/')[-1][1:5]  # image name 0005
            img += '_' + im
            pre = np.zeros((45, 45))  #
            gth = np.zeros((45, 45))

            # ****************** get pck of one image ************************
            target = np.asarray(label_map[b, t, :, :, :].data)  # 3D numpy 21 * 45 * 45
            predict = np.asarray(predict_heatmaps[t][b, :, :, :].data)  # 3D numpy 21 * 45 * 45
            empty = np.zeros((21, 45, 45))

            if not np.equal(empty, target).all():
                pck = PCK(predict, target, sigma=0.04)
                pck_dict[seq + '_' + im] = pck

            # ****************** save image and label of 21 joints ******************
            for i in range(21):  # for each joint
                gth += np.asarray(label_map[b, t, i, :, :].data)  # 2D
                tmp_pre = np.asarray(predict_heatmaps[t][b, i, :, :].data)  # 2D
                pre += tmp_pre

                #  get label of original image
                corr = np.where(tmp_pre == np.max(tmp_pre))
                x = corr[0][0] * (256.0 / 45.0)
                x = int(x)
                y = corr[1][0] * (256.0 / 45.0)
                y = int(y)
                labels_list.append([y, x])  # save img label

            output[0:45, 50 * t: 50 * t + 45] = gth  # save image
            output[50:95, 50 * t: 50 * t + 45] = pre

            label_dict[im] = labels_list  # save label

        # calculate average PCK
        # print pck_dict
        avg_pck = sum(pck_dict.values()) / float(pck_dict.__len__())
        print('step ...%d ... PCK %f  ....' % (step, avg_pck))

        # ****************** save image ******************
        if not os.path.exists(save_dir + 'test'):
            os.mkdir(save_dir + 'test')
        scipy.misc.imsave(save_dir + 'test' + '/s' + str(step) + '_'
                          + seq + img + '_' + str(round(avg_pck, 4)) + '.jpg', output)

        # ****************** save label ******************
        if not os.path.exists(os.path.join(save_dir, 'test_predict')):
            os.mkdir(os.path.join(save_dir, 'test_predict'))

        save_dir_label = os.path.join(save_dir, 'test_predict') + '/' + seq
        if not os.path.exists(save_dir_label):
            os.mkdir(save_dir_label)

        json.dump(label_dict, open(save_dir_label + '/' + str(step) + '.json', 'w'), sort_keys=True, indent=4)
        return pck_dict



from PIL import Image
from PIL import ImageDraw



def draw_point(points, im):
    """
    draw key point on image
    :param points: list 21 [ [x1,y1], ..., [x21,y21]  ]
    :param im: PIL Image
    :return:
    """
    i = 0
    draw=ImageDraw.Draw(im)

    for point in points:
        x = point[1]
        y = point[0]

        if i==0:
            rootx=x
            rooty=y
        if i==1 or i==5 or i==9 or i==13 or i==17:
            prex=rootx
            prey=rooty

        if i >0 and i<=4:
            draw.line((prex,prey,x,y),'red')
            draw.ellipse((x-3, y-3, x+3, y+3), 'red', 'black')
        if i >4 and i<=8:
            draw.line((prex,prey,x,y),'yellow')
            draw.ellipse((x-3, y-3, x+3, y+3), 'yellow', 'black')

        if i >8 and i<=12:
            draw.line((prex,prey,x,y),'green')
            draw.ellipse((x-3, y-3, x+3, y+3), 'green', 'black')
        if i >12 and i<=16:
            draw.line((prex,prey,x,y),'blue')
            draw.ellipse((x-3, y-3, x+3, y+3), 'blue', 'black')
        if i >16 and i<=20:
            draw.line((prex,prey,x,y),'purple')
            draw.ellipse((x-3, y-3, x+3, y+3), 'purple', 'black')


        prex=x
        prey=y
        i=i+1
    return im



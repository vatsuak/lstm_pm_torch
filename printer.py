import numpy as np
import os
import scipy.misc
from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()

#given a series heatmaps, we display the most possible position of the joints
#modification of the save image function in the utils


def pred_images(heatmaps,step,temporal=5, save_dir='./ckpt_tester2'):
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
    b=0
    output = np.ones((50 , 50 * temporal))           # cd .. temporal save a single image
    # seq = imgs[0][b].split('/')[-2]                     # sequence name 001L0
    # img = ""
    for t in range(temporal):                           # for each temporal
        # im = imgs[t][b].split('/')[-1][1:5]             # image name 0005
        # img += '_' + im
        pre = np.zeros((45, 45))  #
        # gth = np.zeros((45, 45))
        for i in range(21):                             # for each joint
            pre += np.asarray((heatmaps[t][b, i, :, :].data).cpu())  # 2D
            # gth += np.asarray((label_map[b, t, i, :, :].data).cpu())        # 2D
        # super_threshold_indices = a > thresh
        # a[super_threshold_indices] = 0
        output[0:45,  50 * t: 50 * t + 45] = pre
        # output[50:95, 50 * t: 50 * t + 45] = pre

        if not os.path.exists(save_dir ):
            os.mkdir(save_dir )
    scaler_model.fit(output)
    output = scaler_model.transform(output)
    scipy.misc.imsave(save_dir + '/' + str(step) + '.jpg', output[0:44])
    return output
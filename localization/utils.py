import os
import cv2
import numpy as np


'''
filename_path = r'F:\kakaresults\one-stage_filename.npy'
dis_path = r'F:\kakaresults\one-stage_result.npy'
filename = np.load(filename_path)
distance = np.load(dis_path)
error = np.array(np.where(distance<30)).squeeze()
'''


def get_datalist(fold=0,error_mode=False):
    
    data_root_path = r"./data/FoveaDetection_Resized"
    split_list_file = os.path.join(data_root_path, "FoveaLoc_Resized.csv")
    img_save_path = os.path.join(data_root_path, "ResizedImages")
    with open(split_list_file) as flist:
        train_file_list = [os.path.join(img_save_path, line.strip()) for line in flist]
    

    trainlist = []
    vallist = []
    for i in range(1,len(train_file_list)):
        if int(os.path.basename(train_file_list[i]).split('_')[0]) % 10 == fold:
            vallist.append(train_file_list[i])
        else:
            trainlist.append(train_file_list[i])
    

    return trainlist, vallist


def check_mkdirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr, decrease_epoch):
    """Sets the learning rate """
    lr = lr * (0.5 ** (epoch // decrease_epoch))
    if lr < 1e-6:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calc_dist(pred_hm, points, ratio=1.0):
    dis_arr = []
    pred_idx = []
    for point, single_pred_hm in zip(points, pred_hm):
        label_y, label_x = point
        pred_x, pred_y = np.unravel_index(single_pred_hm.argmax(), single_pred_hm.shape)
        pred_x, pred_y = pred_x * ratio, pred_y * ratio
        dis_arr.append(np.sqrt((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2))
        pred_idx.append((pred_y, pred_x))

    return np.array(dis_arr), pred_idx


def hm2loc(pred_hm):
    pred_x1, pred_y1 = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
    pred_hm = pred_hm - pred_hm.max() / 3 * 2
    pred_hm[pred_hm < 0] = 0
    pred_hm[pred_hm > 0] = 255
    pred_hm = pred_hm.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    pred_hm = cv2.morphologyEx(pred_hm, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(pred_hm, 1, 2)
    if len(contours) != 1:
        #print('contours: %d' % len(contours))
        return pred_x1, pred_y1
    else:
        cnt = contours[0]
        M = cv2.moments(cnt)
        cx = (M['m10'] / M['m00'])
        cy = (M['m01'] / M['m00'])
        return cy, cx

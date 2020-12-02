# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:08:10 2020

@author: nikoyuan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:31:37 2020

@author: nikoyuan
"""

import csv
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from opts import parse_opts
from datasets_base import vflip_image
from models import get_model
from utils import hm2loc


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

models_name_list = [
    'results/fold_1_get_efficientunet_b5result2020_11_2520_51_40_2.7883/2.7883.pkl',
    'results/fold_2_get_efficientunet_b5result2020_11_2520_52_50_2.5962/2.5962.pkl',
    'results/fold_3_get_efficientunet_b5result2020_11_2610_12_42_2.6584/2.6584.pkl',
    'results/fold_4_get_efficientunet_b5result2020_11_2610_13_22_2.7748/2.7748.pkl',
    'results/fold_5_get_efficientunet_b5result2020_11_2616_19_27_2.9805/2.9805.pkl',
    'results/fold_6_get_efficientunet_b5result2020_11_2720_01_41_2.7838/2.7838.pkl',
    'results/fold_7_get_efficientunet_b5result2020_11_3009_48_29_2.9598/2.9598.pkl',
    'results/fold_8_get_efficientunet_b5result2020_11_3009_49_32_2.7152/2.7152.pkl',
    'results/fold_9_get_efficientunet_b5result2020_11_3009_50_44_2.6375/2.6375.pkl',
    'results/fold_0_get_efficientunet_b5result2020_11_3009_52_02_2.5169/2.5169.pkl'
]




import ttach as tta
transforms_tta = tta.Compose(
    [
    #tta.HorizontalFlip(),
    tta.VerticalFlip(),
    tta.Resize(sizes=[[448,448],[512,512],[640,640]],original_size=[512,512])
    ])

class Normalize(object):
    def __init__(self, mean, std):
        '''
        :param mean: RGB order
        :param std:  RGB order
        '''
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)

    def __call__(self, image):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        # plt.figure(1)
        # plt.imshow(image)
        # plt.show()
        return (image.transpose((2, 0, 1)) - self.mean) / self.std


def predict_image(model, img_crop):
    
    with torch.no_grad():   
        img_crop = torch.from_numpy(img_crop.transpose(2, 0, 1)).float().unsqueeze(0).cuda()#.to(device)
        tta_model = tta.SegmentationTTAWrapper(model, transforms_tta)
    
        img_out = tta_model(img_crop)[0][0].cpu().detach().numpy()

    return img_out




def main():
    opt = parse_opts()
    #device = torch.device("cuda:%d" % 0)

    models_list = []
    for i in range(len(models_name_list)):
        model = get_model(opt)
        print('Model Loading: '+str(models_name_list[i]))
        model.load_state_dict((torch.load(models_name_list[i])))
        model.cuda()
        model.eval()
        #model = torch.nn.DataParallel(model)
        models_list.append(model)

    data_root_path = r"./data/Refuge2-Validation"
    coarse_label_list = os.listdir(data_root_path)
    
    results = []
    for i in tqdm(range(len(coarse_label_list))):
        file = coarse_label_list[i]
        
        img = cv2.imread(os.path.join(data_root_path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        ratio = img.shape[1] / 512
        ratio2 = img.shape[0] / 512
        if ratio != ratio2:
            print('WARNING: the size is not equal.')
        img = cv2.resize(img, (512, 512))
        
        out_pred_x_all = []
        out_pred_y_all = []
        for model in models_list:
           
            img_out = predict_image(model, img)
            l_pred_y, l_pred_x = hm2loc(img_out)

            out_pred_x = (l_pred_x ) * ratio
            out_pred_y = (l_pred_y ) * ratio2
            out_pred_x_all.append(out_pred_x)
            out_pred_y_all.append(out_pred_y)

        results.append([file, np.mean(out_pred_x_all), np.mean(out_pred_y_all)])


    output_file = "./outputs/Localization_Results_base_ensemble_tta_all_origin_1130.csv"
    with open(output_file, "w+") as f:
        f.write("{},{},{}\n".format("ImageName", "Fovea_X", "Fovea_Y"))
        for result in results:
            f.write("{},{},{}\n".format(result[0], result[1], result[2]))


if __name__ == '__main__':
    main()

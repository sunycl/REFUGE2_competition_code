# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:22:48 2020

@author: nikoyuan
"""

import os
import time
import cv2
import numpy as np
import torch
from datasets_base import DATASET, DATASET_VAL
from models import get_model
from opts import parse_opts
from radam import RAdam
from utils import adjust_learning_rate, AverageMeter, check_mkdirs, get_datalist, hm2loc
from torch import nn
import matplotlib.pyplot as plt
import scipy.io as sio  
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def main():
    opt = parse_opts()
    opt.inputsize = [int(item) for item in opt.input_list.split(',')]
    torch.cuda.set_device(opt.gpuid)
    device = torch.device("cuda:%d" % opt.gpuid)
    folder = 'fold_%d_%sresult' % (opt.fold, opt.model) + time.strftime("%Y_%m_%d%H_%M_%S", time.localtime())
    save_path = os.path.join(opt.result_path, folder)
   
    model = get_model(opt)
    model.to(device)
    trainlist, vallist = get_datalist(opt.fold)
    trainset = DATASET(trainlist)
    valset = DATASET_VAL(vallist)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=7, num_workers=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=2, num_workers=6, shuffle=False)
    optimizer = RAdam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_meandice = 100


    for epoch in range(50):
        adjust_learning_rate(optimizer, epoch, 3e-4, 5)
        train(model, optimizer, train_loader, epoch, 50, device)
        
        mean_dist = val(model, val_loader, epoch, device)
        if mean_dist < best_meandice:
            best_meandice = mean_dist
            check_mkdirs(save_path)
            
            if epoch > 1:
                print('save model...:'+ os.path.join(save_path, '%.4f.pkl' % ( best_meandice)))
                check_mkdirs(save_path)
                torch.save(model.state_dict(), os.path.join(save_path, '%.4f.pkl' % (best_meandice)))
        print('Best Mean Dice: %.4f' % best_meandice)
        
    os.rename(save_path, save_path + '_%.4f' % best_meandice)



def train(model, optimizer, train_loader, epoch, trainepochs, device):
    losses = AverageMeter() 
    model.train()
    loss_fn = torch.nn.MSELoss()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        output = model(data)
        mseloss = loss_fn(target, output) * 100
               
        loss = mseloss 
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 10 == 0:
            log = ('Train Epoch:[{}/{}({:.1f}%)]\t[{:03d}/{:03d}\tLoss: {:.4f}({:.4f})'
                .format(
                epoch, trainepochs, 100. * epoch / trainepochs, batch_idx, len(train_loader),
                losses.val, losses.avg))
            print(log)


def val(model, val_loader, epoch, device):
    model.eval()
    dists = AverageMeter()
    dists2 = AverageMeter()

    for batch_idx, (data, p_x, p_y) in enumerate(val_loader):
        with torch.no_grad():
            data = data.to(device)
            output = model(data)
            output = output.data.cpu().numpy()
            p_x, p_y = p_x.numpy(), p_y.numpy()
            for i in range(output.shape[0]):
                pred_hms = output[i][0]
                pred_x, pred_y = np.unravel_index(pred_hms.argmax(), pred_hms.shape)
                dist = np.sqrt((pred_x - p_y[i]) ** 2 + (pred_y - p_x[i]) ** 2)
                dists.update(dist, 1)
            
            for i in range(output.shape[0]):
                pred_hms = output[i][0]
                pred_x, pred_y = hm2loc(pred_hms)
                dist = np.sqrt((pred_x - p_y[i]) ** 2 + (pred_y - p_x[i]) ** 2)
                dists2.update(dist, 1)
     
    print('The first method: ', epoch, dists.avg)
    print('The second method: ',epoch, dists2.avg)
    
    return dists2.avg


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:15:18 2020

@author: nikoyuan
"""

import cv2

import numpy as np
import torch
from torch.utils.data import Dataset
import datetime
import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def np2tensor(data):
    return torch.from_numpy(data.copy().astype(np.float)).float().unsqueeze(0)


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.001, 0.001),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-20, 20), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                      0,))
    ''' 
    if np.random.random() < u:
        elastic = ElasticTrans(image, mask)
        image, mask = elastic.elastic_transform()
    
    if np.random.random() < u:
        image = random_noise(image, mode='speckle', clip=False)
     '''
        
      
    return image, mask


def vflip_image(image):
    return cv2.flip(image, flipCode=1)



def gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)  ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  ## (height,1)
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    #return np.exp(-(np.sqrt((x - x0) ** 2 + (y - y0) ** 2)) / (2 * sigma))


def generate_hm(height, width, point, s=17):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        point    : (x,y)
    """
    hm = gaussian_k(point[0], point[1], s, height, width)
    return hm

from math import exp, log, sqrt, ceil
from scipy.stats import multivariate_normal
def cal_sigma(dmax, edge_value):
    return sqrt(- pow(dmax, 2) / log(edge_value))

def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)
  
def draw_heatmap(width, height, x, y, sigma=17):
    
    x_ = np.arange(width, dtype=np.float)
    y_ = np.arange(height, dtype=np.float)
    xx, yy = np.meshgrid(x_,y_)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    m1 = (x, y)
    s1 = np.eye(2) * pow(sigma, 2)
    # k1 = multivariate_normal(mean=m1, cov=593.109206084)
    k1 = multivariate_normal(mean=m1, cov=s1)
    #     zz = k1.pdf(array_like_hm)
    zz = gaussian(xxyy.copy(), m1, sigma)
    img = zz.reshape((height,width))
    return img



def test(width, height, x, y, array_like_hm):
    dmax = 100
    edge_value = 0.01
    sigma = cal_sigma(dmax, edge_value)
    
    return draw_heatmap(width, height, x, y, sigma, array_like_hm)



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




class DATASET(Dataset):
    def __init__(self, datalist):
        self.trainlist = datalist

    def __len__(self):
        return len(self.trainlist)

    def __getitem__(self, idx):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        sample = self.trainlist[idx].split(",")
        #sample = trainlist[idx].split(",")
        file, p_x, p_y = sample
        p_x, p_y = float(p_x), float(p_y)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img[:, :, ::-1].astype('float32') / 255.0
        img = img.astype('float32') / 255.0
        #ratio = 1
        #p_x, p_y = p_x * ratio, p_y * ratio
        #img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

        #img, p_x, p_y = crop(img, p_x, p_y)

        if np.random.randint(0, 2): 
            img = vflip_image(img)
            p_x = 512 - p_x - 1
        
 
        hm = generate_hm(512, 512, (p_x, p_y))
        #hm = draw_heatmap(288, 384, p_x, p_y)
        
        img, label = randomShiftScaleRotate(img, hm,
                                            shift_limit=(-0.0, 0.0),
                                            scale_limit=(-0.1, 0.1),
                                            rotate_limit=(-15.0, 15.0),
                                            aspect_limit=(-0.0, 0.0),
                                            borderMode=cv2.BORDER_CONSTANT, u=0.5)
        return torch.from_numpy(img.transpose(2, 0, 1)), np2tensor(label)


class DATASET_VAL(Dataset):
    def __init__(self, valist):
        self.valist = valist

    def __len__(self):
        return len(self.valist)

    def __getitem__(self, idx):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        sample = self.valist[idx].split(",")
        file, p_x, p_y = sample
        p_x, p_y = float(p_x), float(p_y)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img[:, :, ::-1].astype('float32') / 255.0
        img = img.astype('float32') / 255.0

        return torch.from_numpy(img.transpose(2, 0, 1)).float(), p_x, p_y, _, _, _

class DATASET_VAL1(Dataset):
    def __init__(self, valist):
        self.valist = valist

    def __len__(self):
        return len(self.valist)

    def __getitem__(self, idx):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        sample = self.valist[idx].split(",")
        file, p_x, p_y = sample
        p_x, p_y = float(p_x), float(p_y)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img[:, :, ::-1].astype('float32') / 255.0
        img = img.astype('float32') / 255.0

        return torch.from_numpy(img.transpose(2, 0, 1)).float(), p_x, p_y, _, _, _, file


import numpy.linalg as nl
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom
import torchvision.transforms as transforms
from skimage.util import random_noise
from PIL import Image, ImageOps

class ElasticTrans(object):
    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.GRID = 5
        self.SIGMA = 3


    def make_t(self,cp):
        t = squareform(pdist(cp, metric='euclidean'))
        t = t * t
        # a trick to make r ln(r) 0
        t[t == 0] = 1
        t = t * np.log(t)
        np.fill_diagonal(t, 0)
        return t

    def lift_pts(self,p, cp):
        r"""

        Args:
            p (ndarray): input points, size: n x 2
            cp (ndarray): control points, size: k x 2
        Returns:
            p_lift (ndarray): lifted input points: n x (3+k)
        """
        p_lift = cdist(p, cp, 'euclidean')
        p_lift = p_lift * p_lift
        p_lift[p_lift == 0] = 1
        p_lift = p_lift * np.log(p_lift)
        return p_lift

    def tps_estimate(self, source_control_points, target_control_points):
        r""" Estimate the TPS parameters from target to source

        Args:
            source_control_points (ndarray): size: n x 2
            target_control_points (ndarray): size: n x 2
        """

        # construct t
        t = self.make_t(source_control_points)

        # solve cx, cy (coefficients for x and y)
        cx = nl.solve(t, target_control_points[:, 0])
        cy = nl.solve(t, target_control_points[:, 1])
        return cx, cy

    def tps_transform(self,gps, cps, cx, cy):
        # transform [n x k]
        pg_lift = self.lift_pts(gps, cps)
        xgt = np.dot(pg_lift, cx.T)
        ygt = np.dot(pg_lift, cy.T)
        return xgt, ygt

    def map_coord(self, image, coord, order=1):
        if len(image.shape) == 2:
            image_shape = image.shape[::-1]
            return map_coordinates(image, coord, order=order).reshape(image_shape).transpose([1, 0])
        elif len(image.shape) == 3:
            image_shape = image.shape[:-1][::-1] + (1,)
            image_maped = [map_coordinates(image[..., i], coord, order=order).reshape(image_shape).transpose([1, 0, 2]) for i in range(image.shape[-1])]
            return np.concatenate(image_maped, axis=-1)
        else:
            raise ValueError('image dim must be 2 or 3')

    def elastic_transform(self):
        image_size = self.image.shape
        np.random.normal(0, self.SIGMA, 1000)
        sources = []
        targets = []
        for i in range(self.GRID):
            for j in range(self.GRID):
                source_sample = [float(i) / (self.GRID - 1) * image_size[0], float(j) / (self.GRID - 1) * image_size[1]]
                sources.append(source_sample)
                if i == 0 or i == (self.GRID - 1) or i == 0 or i == (self.GRID - 1):
                    targets.append(source_sample)
                else:
                    target_sample = [
                         source_sample[0] + np.random.normal(0, self.SIGMA),
                        source_sample[1] + np.random.normal(0, self.SIGMA),
                    ]
                    targets.append(target_sample)

        cx, cy = self.tps_estimate(np.array(sources), np.array(targets))
        xx, yy = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
        xgs, ygs = xx.flatten(), yy.flatten()
        gps = np.vstack([xgs, ygs]).T
        xgt, ygt = self.tps_transform(gps, np.array(sources), cx, cy)
        indices = np.reshape(xgt, (-1, 1)), np.reshape(ygt, (-1, 1))
        return self.map_coord(self.image, indices, order=1), self.map_coord(self.label, indices, order=1)

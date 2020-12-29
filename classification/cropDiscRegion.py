
import cv2
import numpy as np
from refuge.DiscDetection.models.ResUnet_Nov9 import ResUnet as ResUnet_Disc
from refuge.Tools.BGR2RGB import RGB2BGR, BGR2RGB
import torchvision.transforms as transforms
from refuge.utils import data_transformers
import torch
import torch.autograd as autograd


disc_model = r'./refuge/DiscDetection/models/discSeg_190714_resnet18_256_0.951.h5'  # this one is best

discNet = ResUnet_Disc(resnet='resnet18', num_classes=1)
discNet.load_state_dict(torch.load(disc_model), strict=False)
discNet.cuda()
discNet.eval()

img_size = 256
norm = data_transformers.pytorch_zoo_normaliser
trans_valid = data_transformers.get_test_valid_transformer(img_size, norm)



def cropDiscRegion(Img, cropRatio = 1):
    "cropRatio 1 or segmentatoin ;1.5 / 2 for classification"

    Img1 = transforms.ToPILImage()(Img)
    Img1 = trans_valid(Img1)
    Img1 = autograd.Variable(torch.unsqueeze(Img1, 0).cuda())
    disc_output = discNet(Img1)
    DiscProb_small = disc_output.data.cpu().numpy()[0]

    DiscProb = cv2.resize(DiscProb_small, (Img.shape[1], Img.shape[0]))
    DiscBinaryImg0 = np.uint8(DiscProb > 0)

    Label = DiscBinaryImg0.copy()

    """PostProces"""
    DiscBinaryImg0 = np.uint8(Label > 0 )
    DiscBinaryImg = np.zeros(Img.shape[:2], np.uint8)

    contours_disc, hierarchy = cv2.findContours(DiscBinaryImg0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas_disc = [cv2.contourArea(c) for c in contours_disc]
    max_index_disc = np.argmax(areas_disc)
    cv2.drawContours(DiscBinaryImg, contours_disc, max_index_disc, 255, -1)

    discContour = contours_disc[max_index_disc]

    x, y, w, h = cv2.boundingRect(discContour)
    discX = int(x + w/2)
    discY = int(y + h/2)
    #print(x,y,discX,discY)

    diameter = (w + h) / 2.0 # np.maximum(w, h) #(w + h) / 2.0
    cropRadius = int(cropRatio * diameter) # 1 for segmentatoin ;1.5/2 for classification
    #print(cropRadius)

    borderWidth = cropRadius
    ImgPad = cv2.copyMakeBorder(Img, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT, value=0)
    LabelPad = cv2.copyMakeBorder(Label, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT, value=0)
    xmin_crop = discX + borderWidth - cropRadius
    ymin_crop = discY + borderWidth - cropRadius
    xmax_crop = discX + borderWidth + cropRadius
    ymax_crop = discY + borderWidth + cropRadius
    DiscCrop = ImgPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]
    LabelCrop = LabelPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop]

    return DiscCrop, LabelCrop, borderWidth, xmin_crop, xmax_crop, ymin_crop, ymax_crop


def cropDiscRegion_before(Img, Label):

    DiscBinaryImg0 = np.uint8(Label > 0 )
    DiscBinaryImg = np.zeros(Img.shape[:2], np.uint8)

    contours_disc, hierarchy = cv2.findContours(DiscBinaryImg0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas_disc = [cv2.contourArea(c) for c in contours_disc]
    max_index_disc = np.argmax(areas_disc)
    cv2.drawContours(DiscBinaryImg, contours_disc, max_index_disc, 255, -1)

    discContour = contours_disc[max_index_disc]

    x, y, w, h = cv2.boundingRect(discContour)
    discX = int(x + w/2)
    discY = int(y + h/2)
    #print(x,y,discX,discY)

    diameter = (w + h) / 2.0
    cropRadius = int(1 * diameter) # 1 for segmentatoin ;1.5 for classification
    #print(cropRadius)

    borderWidth = cropRadius
    ImgPad = cv2.copyMakeBorder(Img, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT, value=0)
    LabelPad = cv2.copyMakeBorder(Label, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT, value=0)
    xmin_crop = discX + borderWidth - cropRadius
    ymin_crop = discY + borderWidth - cropRadius
    xmax_crop = discX + borderWidth + cropRadius
    ymax_crop = discY + borderWidth + cropRadius
    DiscCrop = ImgPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]
    LabelCrop = LabelPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop]

    return DiscCrop, LabelCrop, borderWidth, xmin_crop, xmax_crop, ymin_crop,ymax_crop 

def PostProcess_ycl(prediction, ImgShow, Img_step2, borderWidth, xmin_crop, xmax_crop, ymin_crop, ymax_crop, fit=True, enhance=None):
    
    CupDiscProb0_small = prediction.data.cpu().numpy()[0]
    CupPredProb = np.float32(CupDiscProb0_small[0])
    DiscPredProb = np.float32(CupDiscProb0_small[1])
  
    CupDiscPredImg = np.zeros((DiscPredProb.shape[0], DiscPredProb.shape[1]))
    CupDiscPredImg[DiscPredProb > 0.5] = 150
    CupDiscPredImg[CupPredProb > 0.5] = 255

    #print(CupDiscPredImg.shape)
    #print(a)
    CupDiscPredImg = cv2.resize(CupDiscPredImg,(Img_step2.shape[1],Img_step2.shape[0]), interpolation=cv2.INTER_NEAREST)
    # CupPredProb = cv2.resize(CupPredProb, (Img_step2.shape[1], Img_step2.shape[0]))
    # DiscPredProb = cv2.resize(DiscPredProb, (Img_step2.shape[1], Img_step2.shape[0]))

    
    CupDiscPredImg = np.uint8(CupDiscPredImg)
    CupDiscBinaryImg0 = np.zeros((ImgShow.shape[0]+2*borderWidth, ImgShow.shape[1]+2*borderWidth))
    CupDiscBinaryImg0[ymin_crop:ymax_crop, xmin_crop:xmax_crop] = CupDiscPredImg
    CupDiscBinaryImg0 = CupDiscBinaryImg0[borderWidth:borderWidth+ImgShow.shape[0], borderWidth:borderWidth+ImgShow.shape[1]]
    CupDiscSeg_final = CupDiscBinaryImg0

    if fit:
        
        # CupDiscBinaryImg0 = CupDiscBinaryImg0[borderWidth:-borderWidth, borderWidth:-borderWidth]

        CupDiscBinaryImg = np.zeros(ImgShow.shape[:2], dtype=np.uint8)
        CupDiscBinaryImg_ellipseFit = np.zeros(ImgShow.shape[:2], dtype=np.uint8)

        contours_disc, hierarchy = cv2.findContours(np.uint8(CupDiscBinaryImg0 > 0), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        areas_disc = [cv2.contourArea(c) for c in contours_disc]
        max_index_disc = np.argmax(areas_disc)

        cv2.drawContours(CupDiscBinaryImg, contours_disc, max_index_disc, 150, -1)
        disc_elipse = cv2.fitEllipse(contours_disc[max_index_disc])
        cv2.ellipse(CupDiscBinaryImg_ellipseFit, disc_elipse, 150, -1)
        # cv2.drawContours(ImgShow, contours_disc, max_index_disc, (255, 255, 255), 5)
        cv2.ellipse(ImgShow, disc_elipse, (255, 0, 0), 3)

        contours_cup, hierarchy = cv2.findContours(np.uint8(CupDiscBinaryImg0 == 255), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        areas_cup = [cv2.contourArea(c) for c in contours_cup]
        max_index_cup = np.argmax(areas_cup)
        # # cv2.drawContours(PredCup_Step2, contours_cup, max_index_cup, 255, -1)

        cv2.drawContours(CupDiscBinaryImg, contours_cup, max_index_cup, 255, -1)
        cup_elipse = cv2.fitEllipse(contours_cup[max_index_cup])
        cv2.ellipse(CupDiscBinaryImg_ellipseFit, cup_elipse, 255, -1)
        # cv2.drawContours(ImgShow, contours_cup, max_index_cup, (0, 255, 0), 5)
        cv2.ellipse(ImgShow, cup_elipse, (0, 255, 0), 3)

        CupDiscSeg_final = CupDiscBinaryImg_ellipseFit.copy()
    CupDiscSeg_final[CupDiscSeg_final == 150] = 255-128
    CupDiscSeg_final = 255 - CupDiscSeg_final

    return CupDiscSeg_final

        

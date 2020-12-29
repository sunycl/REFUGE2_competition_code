
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from .models import data_transformers
import torch.autograd as autograd

from .BGR2RGB import RGB2BGR
from .ActivationFun import sigmoid
from .models.ResUnet_Nov9 import ResUnet


# disc_model = 'DiscDetection/models/Disc_256.h5'
# cup_model = 'DiscDetection/models/Cup_256.h5'

disc_model = 'DiscDetection/models/discSeg_190714_resnet18_256_0.951.h5'

img_size = 256

discNet = ResUnet(resnet='resnet18', num_classes=1)
discNet.load_state_dict(torch.load(disc_model), strict=False)
discNet.cuda()
discNet.eval()
norm = data_transformers.pytorch_zoo_normaliser
trans_valid = data_transformers.get_test_valid_transformer(img_size, norm)



def discDetection(Img):

    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img1 = transforms.ToPILImage()(Img)
    Img1 = trans_valid(Img1)
    Img1 = autograd.Variable(torch.unsqueeze(Img1, 0).cuda())
    disc_output = discNet(Img1)
    DiscProb_small = disc_output.data.cpu().numpy()[0]

    DiscPredProb = cv2.resize(DiscProb_small, (Img.shape[1], Img.shape[0]))

    DiscBinaryImg = np.zeros(Img.shape[:2], dtype=np.uint8)
    discPixCnt = np.count_nonzero(DiscPredProb > 0)
    if discPixCnt >= 1000*(Img.shape[1]/512.):
        contours_disc, hierarchy = cv2.findContours(np.uint8(DiscPredProb > 0), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
        areas_disc = [cv2.contourArea(c) for c in contours_disc]
        max_index_disc = np.argmax(areas_disc)
        cv2.drawContours(DiscBinaryImg, contours_disc, max_index_disc, 255, -1)
        disc_elipse = cv2.fitEllipse(contours_disc[max_index_disc])

    else:
        disc_elipse = None

    return DiscBinaryImg, disc_elipse


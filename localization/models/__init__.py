from models.unet import Unet5
from models.model1 import NestedUNet
from models.efficientunet import get_efficientunet_b5, get_efficientunet_b3
from .hrnet import HRNet
import torch
#import segmentation_models_pytorch as smp


def get_model(opt):
    if opt.model == 'Unet5':
        model = Unet5(feature_scale=2, n_classes=1, is_deconv=True, in_channels=3,
                      is_batchnorm=True)
    if opt.model == 'NestedUNet':
        model = NestedUNet()
    if opt.model == 'get_efficientunet_b5':
        model = get_efficientunet_b5(out_channels=1, pretrained=False)
    if opt.model == 'get_efficientunet_b3':
        model = get_efficientunet_b3(out_channels=1)
    if opt.model == 'HRNET':
        # from .cfg import _C as config
        # from .cfg import update_config
        # from .cfg import MODEL_EXTRAS
        # update_config(config, None)
        model = HRNet(c=16, nof_joints=1, bn_momentum=0.1)
        # net_dict = model.state_dict()
        # pretrained_dict = torch.load('./weights/pose_hrnet_w48_384x288.pth')
        # pretrain_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.keys()}[:-1]
        # net_dict.update(pretrain_dict)
        # model.load_state_dict(net_dict)
    
    #if opt.model == 'resnet34':
    #    model = smp.Unet('resnet34', encoder_weights='imagenet', activation=None)
    return model

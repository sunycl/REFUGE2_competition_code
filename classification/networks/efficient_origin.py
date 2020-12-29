# encoding: utf-8
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import pretrainedmodels
from torchvision import models
from .efficientnet_pytorch import EfficientNet
from .resnet import resnext50_32x4d 

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path)

    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(load_weights)
    model.load_state_dict(model_weights)

    return model

class EfficientNet_b0(nn.Module):
    def __init__(self, n_class, pre_train = True):
        super(EfficientNet_b0, self).__init__()

        self.model = EfficientNet.from_name('efficientnet-b0')
        #self.model = EfficientNet.from_name('efficientnet-b5')
        #self.model = EfficientNet.from_name('efficientnet-b2')
        #self.model = EfficientNet.from_name('efficientnet-b4') 

        if pre_train:
            self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b0-355c32eb.pth')
            #self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b5-b6417697.pth')
            #self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b2-8bb594d6.pth')
            #self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b4-6ed6700e.pth')

        self.features = self.model.extract_features
        self.num_ftrs = 1280
        #self.num_ftrs = 2048
        #self.num_ftrs = 1408
        #self.num_ftrs = 1792
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier1 = nn.Sequential(nn.Linear(self.num_ftrs, n_class), nn.Sigmoid())
        #self.classifier2 = nn.Linear(self.num_ftrs, n_class)

    def forward(self, x):
        features = self.features(x)
        f = self.GlobalAvgPool(features)
        f = f.view(f.size(0), -1)
        output1 = self.classifier1(f)
        return output1

class SENet154(nn.Module):
    def __init__(self, n_class, pre_train = True):
        super(SENet154, self).__init__()

        if pre_train:
            self.model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained=None)

        self.features = self.model.features
        self.num_ftrs = 2048
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier1 = nn.Sequential(nn.Linear(self.num_ftrs, n_class), nn.Sigmoid())
        #self.classifier1 = nn.Linear(self.num_ftrs, n_class)

    def forward(self, x):
        features = self.features(x)
        f = self.GlobalAvgPool(features)
        f = f.view(f.size(0), -1)
        output1 = self.classifier1(f)
        return output1

class Xception(nn.Module):
    def __init__(self, num_classes, pre_train=True):
        super(Xception, self).__init__()

        if pre_train:
            self.model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained=None)

        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes), nn.Sigmoid())

    def forward(self, x):
        f = self.model.features(x)
        f = self.GlobalAvgPool(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class ResNext50(nn.Module):
    def __init__(self, num_classes, pre_train=False):
        super(ResNext50, self).__init__()

        if pre_train:
            original_model = resnext50_32x4d(True)
        else:
            original_model = resnext50_32x4d(False)
        
        original_model = load_weights(original_model, '/extracephonline/medai_data_tronbian/eye/resnext50_32x4d-7cdf4587.pth')
       

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes), nn.Sigmoid())


    def forward(self, x):
        f = self.features(x)
        f = self.GlobalAvgPool(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class DenseNet121_v0(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, n_class):
        super(DenseNet121_v0, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        self.features = self.densenet121.features
        num_ftrs = self.densenet121.classifier.in_features
        self.classifier1 = nn.Sequential(
            nn.Linear(num_ftrs, n_class),
            nn.Sigmoid()
        )
        self.classifier2 = nn.Linear(num_ftrs, n_class)
        #self.classifier3 = nn.Linear(num_ftrs, num_ftrs)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        output1 = self.classifier1(out)
        output2 = self.classifier2(out)
        return output1, output2


from resnest.torch import resnest50, resnest101, resnest200, resnest269
class ResNest(nn.Module):
    def __init__(self, num_classes, pre_train=True):
        super(ResNest, self).__init__()

        if pre_train:
            original_model = resnest200(pretrained=True)
        else:
            original_model = resnest200(pretrained=False)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes), nn.Sigmoid())


    def forward(self, x):
        f = self.features(x)
        y = self.classifier(f)

        return y

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNest_unc(nn.Module):
    def __init__(self, num_classes, pre_train=True):
        super(ResNest_unc, self).__init__()

        if pre_train:
            original_model = resnest50(pretrained=True)
        else:
            original_model = resnest50(pretrained=False)

        #original_model = load_weights(original_model, '/extracephonline/medai_data_tronbian/eye/resnext50_32x4d-7cdf4587.pth')

        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes), nn.Sigmoid())
        self.classifier2 = nn.Linear(2048, num_classes)
        #self.se = SELayer(2048,64) 

    def forward(self, x):
        #print(x.shape)
        f = self.features(x)
        #f = self.se(f)
        f = self.GlobalAvgPool(f)
        f = torch.flatten(f, 1)
        #print(f.shape)
        y = self.classifier(f)
        y2 = self.classifier2(f)

        return y, y2

class ResNest_unc_mask(nn.Module):
    def __init__(self, num_classes, pre_train=True):
        super(ResNest_unc_mask, self).__init__()

        if pre_train:
            original_model = resnest200(pretrained=True)
        else:
            original_model = resnest200(pretrained=False)

        #original_model = load_weights(original_model, '/extracephonline/medai_data_tronbian/eye/resnext50_32x4d-7cdf4587.pth')
        print()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes), nn.Sigmoid())
        self.classifier2 = nn.Linear(2048, num_classes)
        #self.conv1 =  nn.Conv2d(2048, 1, 1)
        self.upsample_224 = nn.Upsample(size=(224), mode='bilinear')

    def forward(self, x):
        #print(x.shape)
        f_map = self.features(x) # 24,2048,7,7      
        f = self.GlobalAvgPool(f_map)
        f = torch.flatten(f, 1)
        y = self.classifier(f)
        y2 = self.classifier2(f)
        w0 = self.classifier[0].weight[0,:]#.detach()
        w1 = self.classifier[0].weight[1,:]#.detach()
        w0 = w0.unsqueeze(0)
        w0 = w0.unsqueeze(2)
        w0 = w0.unsqueeze(2)
        w1 = w1.unsqueeze(0)
        w1 = w1.unsqueeze(2)
        w1 = w1.unsqueeze(2)
        attention_map0 = self.upsample_224(F.relu(torch.nn.functional.conv2d(f_map, w0)))
        attention_map1 = self.upsample_224(F.relu(torch.nn.functional.conv2d(f_map, w1)))
        ab_min,_ = torch.min(torch.flatten(attention_map0, 1), dim=1)
        ab_min = ab_min.unsqueeze(1)
        ab_min = ab_min.unsqueeze(1)
        ab_min = ab_min.unsqueeze(1)
        ab_max,_ = torch.max(torch.flatten(attention_map0, 1), dim=1)
        ab_max = ab_max.unsqueeze(1)
        ab_max = ab_max.unsqueeze(1)
        ab_max = ab_max.unsqueeze(1)
        #print(ab_max)
        #print(ab_min)
        #print(a)
        attention_map0 = (attention_map0 - ab_min)/(ab_max - ab_min + 1e-10) #F.sigmoid(attention_map0) # / (1 + torch.exp(-100 * (attention_map0 - 0.4)))
        ab_min,_ = torch.min(torch.flatten(attention_map1, 1), dim=1)
        ab_min = ab_min.unsqueeze(1)
        ab_min = ab_min.unsqueeze(1)
        ab_min = ab_min.unsqueeze(1)
        ab_max,_ = torch.max(torch.flatten(attention_map1, 1), dim=1)
        ab_max = ab_max.unsqueeze(1)
        ab_max = ab_max.unsqueeze(1)
        ab_max = ab_max.unsqueeze(1)      
        attention_map1 = (attention_map1 - ab_min)/(ab_max - ab_min + 1e-10)
        attention_map = (attention_map0 + attention_map1) / 2


        return y, y2, attention_map

class ResNest_unc_seg(nn.Module):
    def __init__(self, num_classes, pre_train=True):
        super(ResNest_unc_seg, self).__init__()

        if pre_train:
            original_model = resnest200(pretrained=True)
        else:
            original_model = resnest200(pretrained=False)

        #original_model = load_weights(original_model, '/extracephonline/medai_data_tronbian/eye/resnext50_32x4d-7cdf4587.pth')
        
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes), nn.Sigmoid())
        self.classifier2 = nn.Linear(2048, num_classes)
        self.conv1 =  nn.Conv2d(2048, 512, 3, padding=1)
        self.conv2 =  nn.Conv2d(512, 1, 3, padding=1)
        
        self.upsample_224 = nn.Upsample(size=(224), mode='bilinear')
 
    def forward(self, x):
        #print(x.shape)
        f_map = self.features(x) # 24,2048,7,7      
        f = self.GlobalAvgPool(f_map)
        f = torch.flatten(f, 1)
        y = self.classifier(f)
        y2 = self.classifier2(f)
        seg = self.upsample_224(self.conv2(F.relu(self.conv1(f_map))))

        return y, y2, seg
        


class resnest200_mcs(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, num_classes):
        super(resnest200_mcs, self).__init__()

        num_ftrs = 2048

        A_model = ResNest(num_classes)
        self.featureA = A_model
        self.classA = A_model.features

        B_model = ResNest(num_classes)
        self.featureB = B_model
        self.classB = B_model.features

        C_model = ResNest(num_classes)
        self.featureC = C_model
        self.classC = C_model.features

        self.combine1 = nn.Sequential(
            nn.Linear(num_classes * 4, num_classes),
            nn.Sigmoid()
        )

        self.combine2 = nn.Sequential(
            nn.Linear(num_ftrs * 3, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, y, z):
        x1 = self.featureA(x)
        y1 = self.featureB(y)
        z1 = self.featureC(z)


        x2 = self.classA(x)
        #x2 = F.relu(x2, inplace=True)
        #x2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)
        y2 = self.classB(y)
        #y2 = F.relu(y2, inplace=True)
        #y2 = F.adaptive_avg_pool2d(y2, (1, 1)).view(y2.size(0), -1)
        z2 = self.classC(z)
        #z2 = F.relu(z2, inplace=True)
        #z2 = F.adaptive_avg_pool2d(z2, (1, 1)).view(z2.size(0), -1)

        combine = torch.cat((x2.view(x2.size(0), -1),
                             y2.view(y2.size(0), -1),
                             z2.view(z2.size(0), -1)), 1)
        combine = self.combine2(combine)

        combine3 = torch.cat((x1.view(x1.size(0), -1),
                              y1.view(y1.size(0), -1),
                              z1.view(z1.size(0), -1),
                              combine.view(combine.size(0), -1)), 1)

        combine3 = self.combine1(combine3)

        return x1, y1, z1, combine, combine3

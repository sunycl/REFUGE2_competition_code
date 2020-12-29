import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from progress.bar import Bar
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from utils.trainer_unc_origin import train_step, validation_step, save_output
from utils.metric_origin import compute_metric
from refuge.cropDiscRegion import cropDiscRegion
import pandas as pd
from networks.efficient_origin import EfficientNet_b0, SENet154, Xception, ResNext50, DenseNet121_v0, ResNest, ResNest_unc
import ttach as tta
import natsort
from PIL import Image
import cv2
from eval import eval



transforms_tta = tta.Compose(
    [
        tta.FiveCrops(224,224)
    ]
    )

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)


# Setting parameters
parser = argparse.ArgumentParser(description='EyeQ_dense121')
parser.add_argument('--model_dir', type=str, default='./result_origin_unc_refuge/')
parser.add_argument('--pre_model', type=str, default=True)
parser.add_argument('--save_model', type=str, default='classification_results')

parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--label_idx', type=list, default=['Benign', 'Malignant'])
parser.add_argument('--n_classes', type=int, default=2)
# Optimization options
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch-size', default=20, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--loss_w', default=[1], type=list)

args = parser.parse_args()
args.save_dir = './result_origin_unc_refuge'


# Images Labels
save_file_name = 'FINAL_0910_clf/' + args.save_model + '.csv'
folder = r'/extracephonline/medai_data_tronbian/task/refuge/localization/REFUGE_localization/data/DiscSuccess/'
models_list = [
    # 'result_origin_unc_refuge/mix_mix_mix_9821/DenseNet121_v1_fold0_mix_mix_mix_22.tar',
    # 'result_origin_unc_refuge/mix_mix_mix_9821/DenseNet121_v1_fold0_mix_mix_mix_12.tar',
    'result_origin_unc_refuge/DenseNet121_v1_fold3_mix.tar', #0.9707
    'result_origin_unc_refuge/DenseNet121_v1_fold2_mix.tar', #0.9628
    'result_origin_unc_refuge/DenseNet121_v1_fold10_mix.tar',#0.9625
    'result_origin_unc_refuge/DenseNet121_v1_fold2_modification_0908.tar', #0.9956,
    'result_origin_unc_refuge/DenseNet121_v1_fold1_modification_5_1_0908.tar', #0.9667
    'result_origin_unc_refuge/DenseNet121_v1_fold1_modification_5_2_0908.tar', #0.9631
    'result_origin_unc_refuge/DenseNet121_v1_fold1_modification_5_3_0908.tar' #0.9642
    ]

# options
cudnn.benchmark = True


models = []
for i in range(len(models_list)):

    model = ResNest_unc(args.n_classes)  
    loaded_model = torch.load(models_list[i])
    model.load_state_dict(loaded_model['state_dict'])
    model.to(device)
    model.eval()
    models.append(model)


criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))


transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(256)
    ])



# Testing
outPRED_mcs = torch.FloatTensor().cuda()


imgList = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames
        if (  # os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.PNG' or
                os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.JPG'
                or os.path.splitext(f)[1] == '.bmp' or os.path.splitext(f)[1] == '.BMP')]
ImgList = [x[len(folder):] for x in imgList]

ImgList = natsort.natsorted(ImgList)
print(len(ImgList), ImgList[:2])
iters_per_epoch = len(ImgList)
bar = Bar('Processing {}'.format('inference'), max=len(ImgList))
bar.check_tty = False



for ImgNumber in  range(len(ImgList)):

    begin_time = time.time()
    imgName = folder  + ImgList[ImgNumber]
    Img0 = cv2.imread(imgName)
    Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
    ImgShow = Img.copy()

    ##############################################################
    """Preprocess: Crop Disc Region"""
    
    DiscCrop, _, _, _, _, _, _ = cropDiscRegion(Img, cropRatio=1.5) #1 for seg; 1.5/2 for class
    DiscCrop = Image.fromarray(DiscCrop).convert('RGB')
    DiscCrop = transformList2(np.array(transform_list_val1(DiscCrop)).astype('float32'))
    imagesA = torch.FloatTensor(DiscCrop).cuda()
    imagesA = imagesA.unsqueeze(0)

    result_mcs_all_ensemble = []
    for j in range(len(models)):
        result_mcs_all = []
        for transformer in transforms_tta: # custom transforms or e.g. tta.aliases.d4_transform() 

            imagesA = transformer.augment_image(imagesA).cuda()
            #imagesA = imagesA.cuda()
            result_mcs,_= models[j](imagesA)
            result_mcs_all.append(result_mcs.cpu().detach().numpy())

        result_mcs = torch.from_numpy(np.mean(np.array(result_mcs_all),0)).cuda()
        result_mcs_all_ensemble.append(result_mcs.cpu().detach().numpy())
    result_mcs = torch.from_numpy(np.mean(np.array(result_mcs_all_ensemble),0)).cuda()   

    outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
    batch_time = time.time() - begin_time
    bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(ImgNumber + 1, len(ImgList),
                                                        batch_time=batch_time * (iters_per_epoch - ImgNumber) / 60)
    bar.next()
bar.finish()

# save result into excel:

outPRED_mcs = outPRED_mcs.cpu().detach().numpy()
with open(save_file_name, "w+") as f:
    f.write("{},{}\n".format("FileName", "Glaucoma Risk"))
    for i in range(len(outPRED_mcs)):
        f.write("{},{}\n".format(ImgList[i], outPRED_mcs[i][1]))




    


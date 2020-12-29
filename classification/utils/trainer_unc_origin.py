import time
import torch
from progress.bar import Bar
import numpy as np
import pandas as pd

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
    """
    
    assert v.dim() == 2
    n, c = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n *  np.log2(c))


def train_step(train_loader, model, epoch, optimizer, criterion, args):

    # switch to train mode
    model.train()
    epoch_loss = 0.0
    loss_w =args.loss_w

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)

        imagesA = imagesA.cuda()
     
        labels = labels.cuda()
        labels = labels[:,0,:]  

        out_A,  out_std_A  = model(imagesA)
       
        loss_x = criterion[0](out_A, labels)
        
        weight_out_std_A = torch.sqrt(torch.exp(out_std_A))

        weight_factor = 0.5
        loss_x  = torch.mean(loss_x * (1 + weight_factor * weight_out_std_A))

        # entropy loss
        #weight_entropy = 0.001

        #entropy_loss_x =  entropy_loss(out_A) * weight_entropy * ( 1 + torch.sum(weight_out_std_A))

        # unc loss
        weight_unc_factor = 0.001
        uncloss_x = criterion[1](out_A, out_std_A, labels) *  weight_unc_factor

        lossValue = 1 * (loss_x+uncloss_x)
        #lossValue = (lossValue-0.2).abs() + 0.2

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
    

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} '
        bar.suffix = bar_str.format(step+1, iters_per_epoch, batch_time=batch_time*(iters_per_epoch-step)/60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return epoch_loss


def validation_step(val_loader, model, criterion):

    # switch to train mode
    model.eval()
    epoch_loss = 0
    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(val_loader):
        start_time = time.time()

        imagesA = imagesA.cuda()
        labels = labels.cuda()
        labels = labels[:,0,:]

        outputs, _ = model(imagesA)
        with torch.no_grad():
            loss = torch.mean(criterion[0](outputs, labels))
            epoch_loss += loss.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    bar.finish()
    return epoch_loss


def save_output(label_test_file, dataPRED, args, save_file):
    label_list = args.label_idx
    n_class = len(label_list)
    datanpPRED = np.squeeze(dataPRED.cpu().numpy())
    df_tmp = pd.read_csv(label_test_file)
    image_names = df_tmp["ID"].tolist()

    result = {label_list[i]: datanpPRED[:, i] for i in range(n_class)}
    result['ID'] = image_names
    out_df = pd.DataFrame(result)

    name_older = ['ID']
    for i in range(n_class):
        name_older.append(label_list[i])
    out_df.to_csv(save_file, columns=name_older)

import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
class Uncloss(nn.Module):
    def __init__(self):
        super(Uncloss, self).__init__()

    def forward(self, inputs, intputs_std, targets):
            #box_regression = F.softmax(inputs,-1)
            box_regression = inputs
            box_std_regression = intputs_std
            #box_std_regression =  F.softmax(intputs_std, -1)
            #regression_targets = torch.FloatTensor(targets.size()[0], 6).cuda()
            #regression_targets.zero_()
            #regression_targets.scatter_(1, targets.unsqueeze(1).long().cuda(), 1)
            regression_targets = targets
            # xe - xg
            bbox_in = box_regression - regression_targets
            # print('bbox_in: ', bbox_in)
            bbox_inside_weights = torch.ones_like(bbox_in)
            bbox_inw = torch.mul(bbox_in, bbox_inside_weights)
            bbox_l1abs = torch.abs(bbox_inw)
            bbox_sq = torch.mul(bbox_l1abs, bbox_l1abs)

            wl2 = torch.lt(bbox_l1abs, torch.ones_like(bbox_l1abs))

            #wl1 = wl1.float()
            wl2 = wl2.float()
            bbox_l2 = torch.mul(torch.mul(bbox_sq, wl2),0.5) #boxl2 = 1/2 (xg-xe)^2 x<=1
            #bbox_l1 = torch.mul((bbox_l1abs - 0.5), wl1) #boxl1 = (|xg-xe|-0.5)
            #bbox_inws = bbox_l1  + bbox_l2
            bbox_inws = bbox_l2
            # print('box_inws: ',bbox_inws)

            bbox_pred_std_abs = torch.mul(box_std_regression,0.5) #a/2
            bbox_pred_std_nabs = -1 * box_std_regression #-a
            bbox_pred_std_nexp = torch.exp(bbox_pred_std_nabs) #e^(-a)
            bbox_inws_out = torch.mul(bbox_pred_std_nexp, bbox_inws)

            # loss 1
            bbox_pred_std_abs_logw = torch.mul(bbox_pred_std_abs, torch.ones_like(bbox_inws_out)) #a/2
            bbox_pred_std_abs_logwr = torch.mean(bbox_pred_std_abs_logw, dim=0) #squeeze batch
            bbox_pred_std_abs_logw_loss = torch.sum(bbox_pred_std_abs_logwr) #uncer loss
            # loss 2
            bbox_inws_outr = torch.mean(bbox_inws_out, dim=0)
            bbox_pred_std_abs_mulw_loss = torch.sum(bbox_inws_outr)

            #print('bbox_pred_std_abs_logw_loss: ',bbox_pred_std_abs_logw_loss/ inputs.size()[0] )
            #print('bbox_pred_std_abs_mulw_loss: ',bbox_pred_std_abs_mulw_loss/ inputs.size()[0])
            #cls_weight = torch.sqrt(torch.exp(box_std_regression))
            return  (bbox_pred_std_abs_logw_loss + bbox_pred_std_abs_mulw_loss) / inputs.size()[0]


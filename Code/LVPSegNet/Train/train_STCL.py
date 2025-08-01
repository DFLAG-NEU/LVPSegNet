
import sys
import argparse
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from Dataset import BasicDataset_2D, BasicDataset_3D
from Sampler import CLSampler
# 将DS的功能全部整合
from Image_process import cal_centerofmass, cal_cropbox, normalization
from Loss import masked_CE_loss, masked_dice_loss, masked_multiclass_dice_coeff, CE_loss, \
    dice_loss, B_dice_coeff3D

from Model.STSN import kiSTSN_plus
from Utils.show import curve_show
import torch.nn.functional as F
from pymic.util.parse_config import parse_config
import shutil
import os

import math


def init_model(type='ki_plus', fusion_flag=True, softmax_flag=True,direction='L2H',fusion_mode=0):
    if type == 'ki_plus':
        return kiSTSN_plus(class_num=4,softmax_flag=softmax_flag,fusion_flag=fusion_flag,direction=direction,fusion_mode=fusion_mode)
    else:
        return None

def cal_loss(pred, true, roi=None, slice_based_flag=False, CE_flag=True):
    n_classes = 4
    # 有可能考量只用一种DSC
    if roi is None:
        CEL = CE_loss(softmax_flag=False)
        DSCL = dice_loss
        loss=0
        loss = DSCL(pred.float(), F.one_hot(true, n_classes).permute(0, 3, 1, 2).float(), multiclass=True,
                    reduce_batch_first=(1 - slice_based_flag))
        if CE_flag:
            loss += CEL(pred.float(), true, slice_based_flag=slice_based_flag)

    else:
        # 是否加mask

        CEL = masked_CE_loss(softmax_flag=False)
        DSCL = masked_dice_loss
        loss = DSCL(pred.float(), F.one_hot(true, n_classes).permute(0, 3, 1, 2).float(),
                    ((roi.unsqueeze(1)).repeat(1, n_classes, 1, 1)).float(), multiclass=True,
                    reduce_batch_first=(1 - slice_based_flag))
        if CE_flag:
            loss += CEL(pred.float(), true, roi.float(), slice_based_flag=slice_based_flag)
    return loss


def Train(net,
          n_classes,
          train_dataloader,
          train_dataloader_spl,
          val_2Ddataloader,
          val_dataloader,
          out,
          nEpochs,
          LR,
          cuda,
          net_crop_size=128,
          threshold=30,
          norm_flag=True,
          out_file='',
          slice_based_flag=True,
          roi_loss_flag=True,
          CE_flag=True,
          ep_list=None,
          sm_list=None,
          reverse_flag=0,
          save_sample_id_by_diff=False,
          augCL_flag=False
          ):
    ##all slice based!


    file = open(out_file, 'a')
    optim_net = optim.Adam(net.parameters(), lr=LR)
    # 学习率不做调整了吗，其实不太好

    x = []
    loss_map={'train':[],'val':[]}
    score_map={'r_3D':[],'g_3D':[],'b_3D':[],'mean_3D':[],
               'r_2D': [], 'g_2D': [], 'b_2D': [], 'mean_2D': [],
               }


    max_dice = -1
    max_dice_2D = -1
    min_loss = 999
    earlystop_num = 0

    previous_stage = -1
    diff_list = None
    stage = len(ep_list)
    for epoch in range(nEpochs):

        if augCL_flag:
            aug_rate = np.tanh(epoch / nEpochs)
            train_dataloader.dataset.rate=aug_rate
            #以一种逐渐上升的概率姿态

        s = 0
        for j in range(stage):
            if epoch >= ep_list[j]:
                s += 1
        if not s == previous_stage:
            previous_stage = s
            net.eval()
            with torch.no_grad():
                score_list = np.zeros((len(train_dataloader_spl)))

                for i, data in enumerate(train_dataloader_spl):
                    # BS为1
                    # 此处index就对应文件号码

                    img, label, roi, index = data


                    label = torch.where(label == 4, 0, label)
                    img = img * roi.unsqueeze(1)
                    if cuda:
                        roi = roi.cuda()
                        masks_true = (label.cuda())
                        masks_pred = net((img).cuda())
                    if roi_loss_flag:
                        loss = cal_loss(masks_pred, masks_true, roi, slice_based_flag=slice_based_flag,
                                        CE_flag=CE_flag)
                    else:
                        loss = cal_loss(masks_pred, masks_true, slice_based_flag=slice_based_flag, CE_flag=CE_flag)

                    score = ((loss)).item()



                    score_list[index.numpy()[0]] = score


                if diff_list is None:
                    previous_diff_list=[]
                    new_diff_list=list(range(len(score_list)))
                    #全集

                else:
                    previous_diff_list=diff_list[:train_dataloader.batch_sampler.valid_num]
                    new_diff_list=diff_list[train_dataloader.batch_sampler.valid_num:]


                score_list=score_list[new_diff_list]
                sorted_index = np.argsort(score_list)###index of the image

                diff_list=np.array(new_diff_list)[sorted_index]

                if save_sample_id_by_diff:
                    np.save(out + '/diff_list_S%d.npy' % s, diff_list)

                diff_list = diff_list.tolist()
                #删除一些元素的排序
                if reverse_flag == 1:
                    diff_list = diff_list[::-1]
                diff_list=previous_diff_list+diff_list
                # to save

        net.train()
        mean_loss = 0.0
        train_num = 0

        train_dataloader.batch_sampler.diff_list = diff_list
        train_dataloader.batch_sampler.valid_num = sm_list[s]
        iter_num_valid = (
            math.ceil(train_dataloader.batch_sampler.valid_num / train_dataloader.batch_sampler.batch_size))

        for i, data in enumerate(train_dataloader):
            img, label, roi = data
            if i < iter_num_valid:
                img = img * (roi.unsqueeze(1))
                # 再进行G的训练
                net.train()
                net.zero_grad()
                if cuda:
                    roi = roi.cuda()
                    masks_true = Variable(label.cuda())
                    masks_pred = net(Variable(img).cuda())
                if roi_loss_flag:
                    gen_loss = cal_loss(masks_pred, masks_true, roi, slice_based_flag=slice_based_flag, CE_flag=CE_flag)
                else:

                    gen_loss = cal_loss(masks_pred, masks_true, slice_based_flag=slice_based_flag, CE_flag=CE_flag)
                # 啥情况
                gen_loss.backward()  # 将误差反向传播
                optim_net.step()  # 更新参数
                mean_loss += gen_loss.item() * img.shape[0]
                train_num += img.shape[0]
                sys.stdout.write(
                    '\r[%d/%d][%d/%d]  Train_loss:%f' % (
                        epoch, (nEpochs), i, len(train_dataloader), gen_loss.item()))
        sys.stdout.write(
            '\r[%d/%d][%d/%d]  Train_loss:%f\n' % (
                epoch, (nEpochs), i, len(train_dataloader), mean_loss / train_num))
        file.write(
            '\r[%d/%d][%d/%d]  Train_loss:%f\n' % (
                epoch, (nEpochs), i, len(train_dataloader), mean_loss / train_num))

        loss_map['train'].append((mean_loss / train_num))

        ###看情况进行validation,此处建议直接进行在test上的测试来确定最优的模型,直接进行3Dload，直接求dice score
        if epoch % 1 == 0:
            net.eval()
            print('val')
            mean_fore_3D_DSC,mean_fore_2D_DSC,val_num,mean_val_loss,val_num2D = 0.0,0.0,0,0.0,0
            mean_dice_score_g,mean_dice_score_r,mean_dice_score_b=0,0,0
            with torch.no_grad():
                for i, data in enumerate(val_2Ddataloader):
                    img, label, roi = data
                    img = img * roi.unsqueeze(1)
                    if cuda:
                        roi = roi.cuda()
                        masks_true = Variable(label.cuda())
                        masks_pred = net(Variable(img).cuda())
                    if roi_loss_flag:
                        gen_loss = cal_loss(masks_pred, masks_true, roi, slice_based_flag=slice_based_flag,
                                            CE_flag=CE_flag)
                    else:
                        gen_loss = cal_loss(masks_pred, masks_true, slice_based_flag=slice_based_flag, CE_flag=CE_flag)
                    mean_val_loss += gen_loss.item() * img.shape[0]

                    masks_pred = masks_pred.argmax(dim=1)
                    dice_score_2D = masked_multiclass_dice_coeff(
                        (F.one_hot(masks_pred, n_classes).permute(0, 3, 1, 2).float())[:, 1:, :, :],
                        (F.one_hot(masks_true, n_classes).permute(0, 3, 1, 2).float())[:, 1:, :, :],
                        (((roi.unsqueeze(1)).repeat(1, n_classes, 1, 1)).float())[:, 1:, :, :],
                        reduce_batch_first=False)

                    mean_fore_2D_DSC += dice_score_2D.item() * img.shape[0]
                    val_num2D += img.shape[0]
                for i, data in enumerate(val_dataloader):
                    img, label, roi = data
                    r_label = torch.where(label == 1, 1, 0)
                    b_label = torch.where(label == 3, 1, 0)
                    g_label = torch.where(label == 2, 1, 0)
                    predict = torch.zeros_like(label)
                    bs, X, Y, Z = label.shape
                    assert bs == 1
                    if cuda:
                        masks_trueg = (g_label.cuda())
                        masks_truer = (r_label.cuda())
                        masks_trueb = (b_label.cuda())
                        for z in range(label.shape[3]):
                            slice_roi = roi[0, :, :, z].clone().cpu().numpy()
                            slice_image = img[0, 0, :, :, z].clone().cpu().numpy()
                            if np.sum(slice_roi) == 0:  # 说明此slice中并不含有R&G区域
                                continue
                            midX, midY = cal_centerofmass(slice_roi.copy())
                            startX, startY = cal_cropbox(midX, midY, 336, net_crop_size)

                            tmp = slice_image[startX:startX + net_crop_size, startY:startY + net_crop_size].copy()
                            if norm_flag:
                                tmp = normalization(tmp.copy())
                            slice_image[startX:startX + net_crop_size, startY:startY + net_crop_size] = tmp
                            slice_masked_image = torch.from_numpy(slice_roi * slice_image)
                            slice_masked_image = torch.unsqueeze(slice_masked_image, 0)
                            slice_masked_image = torch.unsqueeze(slice_masked_image, 0).float()

                            slice_masked_image = slice_masked_image[:, :, startX:startX + net_crop_size,
                                                 startY:startY + net_crop_size]

                            masks_pred = net((slice_masked_image).cuda())

                            masks_pred = F.softmax(masks_pred, dim=1)  ####whatever!!!
                            masks_pred = masks_pred.argmax(dim=1)

                            slice_roi = torch.from_numpy(
                                slice_roi[startX:startX + net_crop_size, startY:startY + net_crop_size]).cuda()

                            masks_pred = (masks_pred * torch.where(slice_roi > 0, 1, 0)).cpu()[0]
                            predict[0, startX:startX + net_crop_size, startY:startY + net_crop_size, z] = masks_pred
                    else:
                        print('false!!!')
                    assert bs == 1
                    if cuda:
                        predict = predict.cuda().float()
                        masks_trueg = masks_trueg.float()
                        masks_truer = masks_truer.float()
                        masks_trueb = masks_trueb.float()

                    # 指标测量！！！

                    g_DSC = B_dice_coeff3D((torch.where(predict == 2, 1, 0)).float(), masks_trueg,
                                           reduce_batch_first=False)
                    r_DSC = B_dice_coeff3D((torch.where(predict == 1, 1, 0)).float(), masks_truer,
                                           reduce_batch_first=False)
                    b_DSC = B_dice_coeff3D((torch.where(predict == 3, 1, 0)).float(), masks_trueb,
                                           reduce_batch_first=False)

                    mean_dice_score_g += g_DSC.item() * bs
                    mean_dice_score_r += r_DSC.item() * bs
                    mean_dice_score_b += b_DSC.item() * bs
                    mean_fore_3D_DSC += (g_DSC.item() + r_DSC.item() + b_DSC.item()) * bs

                    val_num += bs
                    # if torch.sum(masks_trueg) > 0:
                    #     #why ,dont understand!!


            sys.stdout.write(
                '\r[%d/%d][%d/%d] val_loss:%f dice_score_2D: %f dice_score_3D(R/G/B/Mean): %f/%f/%f/%f \n' % (
                    epoch, (nEpochs), i, len(val_dataloader), (mean_val_loss / val_num2D), mean_fore_2D_DSC / val_num2D
                    , mean_dice_score_r / (val_num),mean_dice_score_g / (val_num),mean_dice_score_b / (val_num),mean_fore_3D_DSC / (val_num * 3)
                ))
            file.write(
                '\r[%d/%d][%d/%d] val_loss:%f dice_score_2D: %f dice_score_3D(R/G/B/Mean): %f/%f/%f/%f \n' % (
                    epoch, (nEpochs), i, len(val_dataloader), (mean_val_loss / val_num2D), mean_fore_2D_DSC / val_num2D
                    , mean_dice_score_r / (val_num),mean_dice_score_g / (val_num),mean_dice_score_b / (val_num),mean_fore_3D_DSC / (val_num * 3)
                ))
            x.append(epoch)
            loss_map['val'].append((mean_val_loss / val_num2D))
            score_map['g_3D'].append(mean_dice_score_g/val_num)
            score_map['r_3D'].append(mean_dice_score_r / val_num)
            score_map['b_3D'].append(mean_dice_score_b / val_num)
            score_map['mean_3D'].append(mean_fore_3D_DSC / (val_num * 3))
            score_map['mean_2D'].append(mean_fore_2D_DSC/val_num2D)


            # y[1].append(mean_dice_score / val_num)
            # y[2].append(mean_fore_3D_DSC / (val_num * 3))
            # y[3].append((mean_val_loss / val_num2D))
            # y[4].append(mean_fore_2D_DSC / (val_num2D))

            if max_dice < mean_fore_3D_DSC / (val_num * 3):
                max_dice = mean_fore_3D_DSC / (val_num * 3)
                earlystop_num = 0
                torch.save(net.state_dict(), '%s/net' % out + str(epoch) + '.pth')
            else:
                earlystop_num += 1
                if epoch % 100 == 0:
                    torch.save(net.state_dict(), '%s/net' % out + str(epoch) + '.pth')
                else:
                    if min_loss > mean_val_loss / val_num2D:
                        torch.save(net.state_dict(), '%s/net' % out + str(epoch) + '.pth')
                    else:
                        if max_dice_2D < mean_fore_2D_DSC / val_num2D:
                            torch.save(net.state_dict(), '%s/net' % out + str(epoch) + '.pth')
            if min_loss > mean_val_loss / val_num2D:
                min_loss = mean_val_loss / val_num2D
            if max_dice_2D < mean_fore_2D_DSC / val_num2D:
                max_dice_2D = mean_fore_2D_DSC / val_num2D

            if earlystop_num >= threshold:
                print('early stopping!!')

    # curve_show([x,], [y1], ['NP'], 'Epoch', 'Value',title=out+'1',save_path=out+'/Train.png')
    curve_show([x, x, x, x], [loss_map['train'], score_map['mean_3D'], loss_map['val'], score_map['mean_2D']], ['Loss', 'F-DSC-3D', 'val-Loss', 'F-DSC-2D'], 'Epoch', 'Value',
               title=out + '2', save_path=out + '/Val.png')
    curve_show([x, x, x], [score_map['r_3D'], score_map['g_3D'], score_map['b_3D']], ['r_3D', 'g_3D', 'b_3D'], 'Epoch', 'Value',
               title=out + '3', save_path=out + '/Val-score.png')

    file.write(str(np.argmax(score_map['mean_3D'])))
    file.write('best 3D \n')
    file.write(str(np.argmin(loss_map['val'])))
    file.write('best loss \n')
    file.write(str(np.argmax(score_map['mean_2D'])))
    file.write('best 2D \n')
    file.close()

    shutil.copy('%s/net' % out + str(np.argmax(score_map['mean_3D'])) + '.pth', '%s/best_score' % out + '.pth')
    shutil.copy('%s/net' % out + str(np.argmax(score_map['mean_2D'])) + '.pth', '%s/best_score_2D' % out + '.pth')
    shutil.copy('%s/net' % out + str(np.argmin(loss_map['val'])) + '.pth', '%s/best_loss' % out + '.pth')
    flist = os.listdir(out)
    for f in flist:
        if f.startswith('net'):
            os.remove('%s/%s' % (out, f))



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--trainbatchsize', type=int, default=16, help='input batch size')
    parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')

    parser.add_argument('--step_length', type=int, default=3000, help='')
    parser.add_argument('--rate', type=int, default=2, help='')

    parser.add_argument('--sp', type=float, default=0.2, help='start p')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--netWeights', type=str, default='', help="path to net weights (to continue training)")
    parser.add_argument('--datatype', type=str, default='140_v3', help="")

    parser.add_argument('--save_path_base', type=str, default='Record',
                        help="path to net weights (to continue training)")
    parser.add_argument('--LR', type=float, default=0.00001, help='learning rate for generator')
    parser.add_argument('--foldid', type=int, default=1, help='')
    parser.add_argument('--slice_based_flag', type=bool, default=False, help='')
    parser.add_argument('--roi_loss_flag', type=bool, default=False, help='')
    parser.add_argument('--CE_flag', type=bool, default=True, help='')
    parser.add_argument('--dilate_rate', type=int, default=5, help='dilate')
    parser.add_argument('--n_classes', type=int, default=4, help='')
    parser.add_argument('--n_channels', type=int, default=1, help='')
    parser.add_argument('--task', type=int, default=101, help='')
    parser.add_argument('--reverse_flag', type=int, default=1, help='')
    parser.add_argument('--save_sample_id_by_diff', type=int, default=0, help='')
    parser.add_argument('--augCL_flag', type=int, default=0, help='')

    return parser.parse_args()


if __name__ == '__main__':
    # 这个阶段是CL
    opt = get_args()
    slice_based_flag = opt.slice_based_flag
    roi_loss_flag = opt.roi_loss_flag
    dilate_rate = opt.dilate_rate
    Rshift_dis = dilate_rate
    CE_flag = opt.CE_flag
    reverse_flag = opt.reverse_flag
    norm_flag = True
    n_classes = opt.n_classes
    n_channels = opt.n_channels
    exp_list = []

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


            #select
    #数据路径

    trainlabeldir = '../Data/S2/train_ideal/label'
    trainimagedir = '../Data/S2/train_ideal/image'
    trainroidir = '../Data/S2/train_ideal/roi'

    vallabeldir = '../Data/S2/val/label'
    valimagedir = '../Data/S2/val/image'
    valroidir = '../Data/S2/val/roi'

    val3Dlabeldir='../Data/np/val/label3D'
    val3Dimagedir = '../Data/np/val/image3D'
    val3Droidir = '../Data/S2/val/roi3D'


    train_crop_flag = False

    step_length = opt.step_length
    # train_DL_list = []

    if not opt.reverse_flag==3:
        sp = opt.sp
    else:
        sp=1

    total_iter = opt.nEpochs * math.ceil(len(os.listdir(trainimagedir)) / opt.trainbatchsize)
    sm_list, ep_list, stage, total_epoch, tmp = [], [], 1, 0, sp
    while tmp < 1:
        tmp *= opt.rate
        stage += 1

    if (not os.path.isfile('CONFIG/C%d.cfg' % opt.task)):
        print("configure file does not exist: {0:} ".format(opt.task))
        exit()
    config = parse_config('CONFIG/C%d.cfg' % opt.task)
    fusion_flag = config['network']['fusion_flag']
    softmax_flag = config['network']['softmax_flag']
    direction = config['network']['direction']
    fusion_mode=config['network']['fusion_mode']
    RPB_flag = config['dataset']['rpb_flag']
    net_crop_size = config['dataset']['net_crop_size']
    cp_out = '%s/%s_fold%d' % (opt.save_path_base+'/'+opt.datatype, opt.task, opt.foldid)

    # 还是按照原来的模式?到底要按照那种CL形式？

    total_len = len(os.listdir(trainimagedir))
    for i in range(stage):
        if i == stage - 1:
            total_epoch += math.ceil((total_iter - (step_length * (stage - 1))) / math.ceil(
                len(os.listdir(trainimagedir)) / opt.trainbatchsize))
            sm_list.append(total_len)

        else:
            p = sp * pow(opt.rate, i)
            # 在这里改变形式
            sample_num = int(total_len * p)
            sm_list.append(sample_num)
            # 此处一共有多少样本
            # 直接调取全集
            total_epoch += math.ceil(step_length / (math.ceil(sample_num / opt.trainbatchsize)))
        ep_list.append(total_epoch)

    train_sampler = CLSampler(opt.trainbatchsize, diff_list=None, valid_num=None)


    if reverse_flag == 1:
        cp_out = cp_out + '_RSPL'

    try:
        os.makedirs(cp_out)
    except OSError:
        pass

    train_dataset = BasicDataset_2D(trainimagedir, trainlabeldir, trainroidir, Rshift_dis=Rshift_dis,
                                    roi_aug_flag=RPB_flag, crop_flag=train_crop_flag, cropsize=net_crop_size)
    val_dataset = BasicDataset_2D(valimagedir, vallabeldir, valroidir, crop_flag=True, roi_aug_flag=False,
                                  cropsize=net_crop_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=opt.workers)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=opt.trainbatchsize,
                                                 num_workers=opt.workers)

    train_dataset_spl = BasicDataset_2D(trainimagedir, trainlabeldir, trainroidir, roi_aug_flag=False,
                                        crop_flag=train_crop_flag, cropsize=net_crop_size, index_flag=True)
    # 检查一下！！
    train_dataloader_spl = torch.utils.data.DataLoader(train_dataset_spl, shuffle=False, batch_size=1,
                                                       num_workers=opt.workers)



    test3D_dataset = BasicDataset_3D(val3Dimagedir, val3Dlabeldir, val3Droidir)
    test3D_dataloader = torch.utils.data.DataLoader(test3D_dataset, shuffle=False, batch_size=1,
                                                    num_workers=opt.workers)

    net = init_model(fusion_mode=fusion_mode,direction=direction, fusion_flag=fusion_flag, softmax_flag=softmax_flag)
    if opt.cuda:
        net.cuda()
    file = open(cp_out + '/record.txt', 'a')
    file.write(str(opt))
    file.write('\n')
    file.close()

    Train(net, n_classes, train_dataloader, train_dataloader_spl, val_dataloader, test3D_dataloader, cp_out,
              total_epoch, opt.LR,
              opt.cuda, norm_flag=norm_flag, out_file=cp_out + '/record.txt', slice_based_flag=slice_based_flag,
              roi_loss_flag=roi_loss_flag, CE_flag=CE_flag, ep_list=ep_list, sm_list=sm_list,
              reverse_flag=reverse_flag, save_sample_id_by_diff=opt.save_sample_id_by_diff,augCL_flag=opt.augCL_flag)

    # 分别列出不同的实验










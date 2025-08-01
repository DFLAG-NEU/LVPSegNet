import numpy as np
import pickle
import nibabel as nib
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import json
from collections import OrderedDict

PATH_preprocessed = '/home/fengteam/lsl/M4D27/nnUnet-OnWindows-main/nnunet/Data/nnUNet_preprocessed'
PATH_raw_data = '/home/fengteam/lsl/M4D27/nnUnet-OnWindows-main/nnunet/Data/nnUNet_raw_data_base/nnUNet_raw_data'
PATH_cropped_data = '/home/fengteam/lsl/M4D27/nnUnet-OnWindows-main/nnunet/Data/nnUNet_raw_data_base/nnUNet_cropped_data'


def mk_dir(cp_out):
    try:
        os.makedirs(cp_out)
    except OSError:
        pass

def load_pickle(path):

    with open(path, 'rb') as f:
        # 从文件中加载数据
        data = pickle.load(f)
    return data


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def copy_files(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dst_folder)
            print(f"Copying {filename} to {dst_folder}")


class CE_loss(nn.Module):
    def __init__(self,  size_average=True,delta=None,softmax_flag=True):
        super(CE_loss, self).__init__()
        self.size_average = size_average
        self.delta = delta
        self.softmax_flag=softmax_flag
    def forward(self, input, target,slice_based_flag=False):#要求target不是one-hot

        # input=torch.clamp(input,1e-4,1-1e-4)
        if input.dim() > 2:  #
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)  # N,H,W

        target = target[:, :]
        input = input[:, :]

        class_num = input.shape[1]
        if self.delta == None:
            delta_list = None
        else:
            delta_list = self.delta
            delta_list = torch.Tensor(delta_list)
        if self.softmax_flag:
            logpt = torch.softmax(input, dim=1)
            # N*H*W,C 先进行softmax 再进行log操作
        else:
            logpt = input
        logpt = torch.clamp(logpt, 1e-4, 1 - 1e-4)
        logpt = torch.log(logpt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        if delta_list is not None:  # 此时代表有权重
            if delta_list.type() != input.data.type():
                delta_list = delta_list.type_as(input.data)
            at = delta_list.gather(0, target.data.view(-1))
            logpt = logpt * at
        loss = -1 * logpt

        if self.size_average:
            return loss.mean()
        else:
            print('False')
def dice_coeff(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    #roi.dim=3
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        #inter=torch.dot(inter.reshape(-1), ROI.reshape(-1))#确立感兴趣区域
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def multiclass_dice_coeff(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):#分不同的channel来进行计算
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...],reduce_batch_first, epsilon)#need to be one hot!!
    return dice / input.shape[1]

def dice_loss(input: Tensor, target: Tensor,reduce_batch_first:bool=True):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff
    return 1 - fn(input, target, reduce_batch_first=reduce_batch_first)

def cal_loss(pred, true, roi=None, slice_based_flag=False, CE_flag=True):
    n_classes = 4
    true=true.long()
    # 有可能考量只用一种DSC
    if roi is None:
        CEL = CE_loss(softmax_flag=False)
        DSCL = dice_loss
        loss = DSCL(pred.float(), F.one_hot(true, n_classes).permute(0, 3, 1, 2).float(),
                    reduce_batch_first=(1 - slice_based_flag))
        if CE_flag:
            loss += CEL(pred.float(), true, slice_based_flag=slice_based_flag)
    return loss


def copy_task(main_task,des_task,index_list):
    #拷贝imagesTs
    #
    #同时拷贝PKL文件

    path_preprocessed='/home/fengteam/lsl/M4D27/nnUnet-OnWindows-main/nnunet/Data/nnUNet_preprocessed'
    path_raw_data='/home/fengteam/lsl/M4D27/nnUnet-OnWindows-main/nnunet/Data/nnUNet_raw_data_base/nnUNet_raw_data'


    #需要修改

    copy_files(src_folder=path_raw_data+'/'+main_task+'/imagesTs',dst_folder=path_raw_data+'/'+des_task+'/imagesTs')

    mk_dir(path_raw_data+'/'+des_task+'/imagesTr')
    mk_dir(path_raw_data+'/'+des_task+'/labelsTr')
    for index in index_list:

        shutil.copy(path_raw_data+'/'+main_task+'/imagesTr/'+index+'_0000.nii.gz',path_raw_data+'/'+des_task+'/imagesTr/'+index+'_0000.nii.gz')
        shutil.copy(path_raw_data+'/'+main_task+'/labelsTr/'+index+'.nii.gz',path_raw_data+'/'+des_task+'/labelsTr/'+index+'.nii.gz')



    #json update

    with open(path_raw_data+'/'+main_task+'/dataset.json', "r", encoding="utf-8") as f:
        content_all= json.load(f)
    content_all['numTraining'] = len(index_list)

    content_all['training'] = []
    train_cases=index_list
    content_all['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
                             in
                             train_cases]

    with open(path_raw_data+'/'+des_task+'/dataset.json', 'w') as f:
        json.dump(content_all, f, indent=4, sort_keys=True)

    ###json update done


    ######pkl update
    splits_ori=load_pickle(path=path_preprocessed+'/'+main_task+'/splits_final.pkl')

    val_keys=splits_ori[0]['val']

    train_keys = []
    train_cases=index_list

    for j in train_cases:
        if j not in val_keys:
            train_keys.append(
                '%s' % j
            )


    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = np.array(train_keys)
    splits[-1]['val'] =val_keys

    mk_dir(path_preprocessed+'/'+des_task)
    pickle_path = path_preprocessed+'/'+des_task+'/splits_final.pkl'
    write_pickle(splits, pickle_path)

    ######pkl update done
    #save
    np.save(path_raw_data+'/'+des_task+'/index.npy',np.array(index_list))
    print('')

def anal_1(rate,pre_task,des_task,main_task='Task901_MIS2'):
    print('')



    path1=PATH_cropped_data+'/'+main_task
    path2=PATH_raw_data+'/'+pre_task+'/inferTr'

    if not os.path.exists(PATH_raw_data+'/'+pre_task+'/index.npy'):
        previous_score_list = None
    else:
        previous_score_list=np.load(PATH_raw_data+'/'+pre_task+'/index.npy').tolist()

    path_preprocessed=PATH_preprocessed
    path_raw_data=PATH_raw_data

    #直接遍历全部

    fset=os.listdir(PATH_raw_data+'/'+main_task+'/imagesTr')
    #删除验证集合


    splits_ori=load_pickle(path=path_preprocessed+'/'+main_task+'/splits_final.pkl')
    val_set=splits_ori[0]['val']

    #直接给一个字典数组得了


    score_list={}


    #val不用遍历，仅遍历train即可


    for f in fset:

        index=f[:f.find('_0000.nii.gz')]


        if index in val_set:
            continue
        if previous_score_list is not None:
            #选择子集
            if index in previous_score_list:
                continue
        assert os.path.exists('%s/%s.pkl' % (path1, index))

        with open('%s/%s.pkl' % (path1, index), 'rb') as f:
            obj_1 = pickle.load(f)
        with open('%s/%s.pkl' % (path2, index), 'rb') as f:
            obj_2 = pickle.load(f)
        assert obj_1['crop_bbox'] == obj_2['crop_bbox']

        data1 = np.load('%s/%s.npz' % (path1, index))
        data1 = data1['data']
        data1 = data1[1]

        data2 = np.load('%s/%s.npz' % (path2, index))
        data2 = data2['softmax']

        assert data1.shape == data2[0].shape

        label = np.where(data1 == 4, 0, data1)
        label=np.where(label<0,0,label)

        masks_true = (torch.from_numpy(label).cuda())
        masks_pred = torch.from_numpy(data2).cuda()
        masks_pred=masks_pred.permute(1,0,2,3)



        loss = cal_loss(masks_pred, masks_true, slice_based_flag=False, CE_flag=True)

        score = ((loss)).item()

        score_list[index] = score





    #进行处理
    #是否有上一阶段的score list



    sr=sorted(score_list.items(),key=lambda x:x[1])
    #from min to max
    sr=sr[::-1]
    #from max to min

    sample_num=int(len(sr)*rate)

    select_sample_list=[]
    for item in sr[:sample_num]:
        select_sample_list.append(item[0])
    for item in val_set:
        if item not in select_sample_list:
            select_sample_list.append(item)

    if previous_score_list is not None:
        for item in previous_score_list:
            if item not in select_sample_list:
                select_sample_list.append(item)
    ######################save the     selecccct sample list

    copy_task(main_task=main_task,des_task=des_task,index_list=select_sample_list)




    ##the score list need to be saved











if __name__ == '__main__':

    stage=3

    #rate calculate

    rate_map={0:(20-0)/(100-0),
              1:(40-20)/(100-20),
              2:(80-40)/(100-40),
              3:(100-80)/(100-80),}

    for fold in [1,2,3,4]:
        pre_task_id='9%d%d'%(fold,stage)
        des_task_id='9%d%d'%(fold,stage+1)
        main_task_id='9%d%d'%(fold,0)

        pre_task='Task%s_MI'%pre_task_id
        des_task='Task%s_MI'%des_task_id
        main_task='Task%s_MI'%main_task_id

        anal_1(rate=rate_map[stage], pre_task=pre_task, des_task=des_task, main_task=main_task)




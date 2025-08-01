import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
def B_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def masked_B_dice_coeff(input: Tensor, target: Tensor, ROI: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        input = torch.mul(input, ROI)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(torch.dot(input.reshape(-1), ROI.reshape(-1))) + torch.sum(torch.dot(target.reshape(-1), ROI.reshape(-1)))
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += masked_B_dice_coeff(input[i, ...], target[i, ...],ROI[i,...])
        return dice / input.shape[0]
def masked_B_dice_loss(input: Tensor, target: Tensor, ROI:Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn =  masked_B_dice_coeff
    return 1 - fn(input[:,0,...], target[:,0,...],ROI[:,0,...], reduce_batch_first=True)

class masked_BCE_loss(nn.Module):
    def __init__(self, size_average=True,weight=[]):
        super(masked_BCE_loss, self).__init__()
        self.size_average = size_average
        if not len(weight)==0:
            if isinstance(weight, list):
                weight = torch.tensor(weight)
            self.weight = weight
        else:
            self.weight=None
    def forward(self, input, target,ROI):#要求target不是one-hot
        # 此时是2分类模式
        input = torch.clamp(input, 1e-4, 1 - 1e-4)
        if input.dim() > 2:  #
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)
        ROI = ROI.view(-1)
        nonZeroRows = torch.abs(ROI) > 0
        target = target[nonZeroRows, :]
        input = input[nonZeroRows, :]

        back_map = torch.where(target == 0, 1, 0)
        back_map = back_map.view(-1)
        input = input.view(-1)
        logpt = torch.where(back_map == 1, torch.log(1 - input), torch.log(input))  #
        target=target.long()
        if not self.weight==None:
            if self.weight.type()!=input.data.type():
                self.weight = self.weight.type_as(input.data)
            at = self.weight.gather(0,target.data.view(-1))
            logpt = logpt *at

        loss = -1*logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def B_dice_coeff3D(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_dice_coeff3D(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def B_recall_g_coeff3D(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        inter = torch.dot(input.reshape(-1), target.reshape(-1))  # 取交集
        sets_sum = torch.sum(target)
        return (inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_recall_g_coeff3D(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def B_precision_g_coeff3D(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        inter = torch.dot(input.reshape(-1), target.reshape(-1))  # 取交集
        sets_sum = torch.sum(input)
        return (inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_precision_g_coeff3D(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def B_recall_g_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)
#此时若绿色像素一个没有，则将返回 recall:1
        return (inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_recall_g_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def B_FN_3D(input: Tensor, target: Tensor, ROI:Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        input = torch.mul(input, ROI)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(torch.dot(target.reshape(-1), ROI.reshape(-1)))
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (sets_sum-inter + epsilon) / (torch.sum(ROI)+ epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_FN_3D(input[i, ...], target[i, ...],ROI[i,...])
        return dice / input.shape[0]
def B_FP_3D(input: Tensor, target: Tensor, ROI:Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        input = torch.mul(input, ROI)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        pred_sum=torch.sum(torch.dot(input.reshape(-1), ROI.reshape(-1)))
        sets_sum = torch.sum(ROI)-torch.sum(torch.dot(target.reshape(-1), ROI.reshape(-1)))
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (pred_sum-inter + epsilon) / (torch.sum(ROI) + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_FP_3D(input[i, ...], target[i, ...],ROI[i,...])
        return dice / input.shape[0]


class masked_CE_loss(nn.Module):
    def __init__(self,  size_average=True,delta=None,softmax_flag=True):
        super(masked_CE_loss, self).__init__()
        self.size_average = size_average
        self.delta = delta
        self.softmax_flag=softmax_flag
    def forward(self, input, target,ROI,slice_based_flag=False):#要求target不是one-hot

        # input=torch.clamp(input,1e-4,1-1e-4)
        if slice_based_flag:
            mean_loss=0
            if input.dim() > 2:  #
                input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target=target.view(target.size(0),-1,1)
            # N,HXW,1
            ROI = ROI.view(ROI.size(0),-1)#N,HXW
            for i in range(ROI.size(0)):#cal with slice

                nonZeroRows = torch.abs(ROI[i]) > 0
                slice_target=target[i][nonZeroRows,:]
                slice_input =input[i][nonZeroRows,:]

                if self.delta == None:
                    delta_list = None
                else:
                    delta_list = self.delta
                    delta_list = torch.Tensor(delta_list)
                if self.softmax_flag:
                    slice_logpt = torch.softmax(slice_input, dim=1)
                    # N*H*W,C 先进行softmax 再进行log操作
                else:
                    slice_logpt = slice_input

                slice_logpt = torch.clamp(slice_logpt, 1e-4, 1 - 1e-4)
                slice_logpt = torch.log(slice_logpt)
                slice_logpt = slice_logpt.gather(1, slice_target)
                slice_logpt = slice_logpt.view(-1)

                if delta_list is not None:  # 此时代表有权重
                    if delta_list.type() != slice_input.data.type():
                        delta_list = delta_list.type_as(slice_input.data)
                    at = delta_list.gather(0, slice_target.data.view(-1))
                    slice_logpt = slice_logpt * Variable(at)
                slice_loss = -1 * slice_logpt
                slice_loss=slice_loss.mean()

                mean_loss+=slice_loss
            return mean_loss/input.size(0)



        else:
            if input.dim() > 2:  #
                input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
            target = target.view(-1, 1)  # N,H,W
            ROI = ROI.view(-1)

            nonZeroRows = torch.abs(ROI) > 0
            target = target[nonZeroRows, :]
            input = input[nonZeroRows, :]

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
                logpt = logpt * Variable(at)
            loss = -1 * logpt

            if self.size_average:
                return loss.mean()
            else:
                print('False')

class CE_loss(nn.Module):
    def __init__(self,  size_average=True,delta=None,softmax_flag=True):
        super(CE_loss, self).__init__()
        self.size_average = size_average
        self.delta = delta
        self.softmax_flag=softmax_flag
    def forward(self, input, target,slice_based_flag=False):#要求target不是one-hot

        # input=torch.clamp(input,1e-4,1-1e-4)
        if slice_based_flag:
            mean_loss=0
            if input.dim() > 2:  #
                input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target=target.view(target.size(0),-1,1)
            # N,HXW,1

            for i in range(target.size(0)):#cal with slice

                slice_target=target[i][:,:]
                slice_input =input[i][:,:]

                if self.delta == None:
                    delta_list = None
                else:
                    delta_list = self.delta
                    delta_list = torch.Tensor(delta_list)
                if self.softmax_flag:
                    slice_logpt = torch.softmax(slice_input, dim=1)
                    # N*H*W,C 先进行softmax 再进行log操作
                else:
                    slice_logpt = slice_input

                slice_logpt = torch.clamp(slice_logpt, 1e-4, 1 - 1e-4)
                slice_logpt = torch.log(slice_logpt)
                slice_logpt = slice_logpt.gather(1, slice_target)
                slice_logpt = slice_logpt.view(-1)

                if delta_list is not None:  # 此时代表有权重
                    if delta_list.type() != slice_input.data.type():
                        delta_list = delta_list.type_as(slice_input.data)
                    at = delta_list.gather(0, slice_target.data.view(-1))
                    slice_logpt = slice_logpt * Variable(at)
                slice_loss = -1 * slice_logpt
                slice_loss=slice_loss.mean()

                mean_loss+=slice_loss
            return mean_loss/input.size(0)


        else:
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
                logpt = logpt * Variable(at)
            loss = -1 * logpt

            if self.size_average:
                return loss.mean()
            else:
                print('False')

def masked_dice_coeff(input: Tensor, target: Tensor, ROI: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    #roi.dim=3
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        masked_input=torch.mul(input,ROI)#无效区域全部乘以0就完事了
        inter = torch.dot(masked_input.reshape(-1), target.reshape(-1))
        #inter=torch.dot(inter.reshape(-1), ROI.reshape(-1))#确立感兴趣区域
        sets_sum = torch.sum(torch.dot(masked_input.reshape(-1), ROI.reshape(-1))) + torch.sum(torch.dot(target.reshape(-1), ROI.reshape(-1)))
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += masked_dice_coeff(input[i, ...], target[i, ...],ROI[i, ...])
        return dice / input.shape[0]
def masked_multiclass_dice_coeff(input: Tensor, target: Tensor, ROI: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):#分不同的channel来进行计算
        dice += masked_dice_coeff(input[:, channel, ...], target[:, channel, ...], ROI[:, channel, ...],reduce_batch_first, epsilon)#need to be one hot!!
    return dice / input.shape[1]
def masked_dice_loss(input: Tensor, target: Tensor, ROI:Tensor,multiclass: bool = False,reduce_batch_first:bool=True):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = masked_multiclass_dice_coeff if multiclass else masked_dice_coeff
    return 1 - fn(input, target,ROI, reduce_batch_first=reduce_batch_first)

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
def dice_loss(input: Tensor, target: Tensor,multiclass: bool = False,reduce_batch_first:bool=True):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=reduce_batch_first)

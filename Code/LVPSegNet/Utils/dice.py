
import torch
from torch import Tensor
import torch.nn as nn
#3d val
def dice_coeff3D(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
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
            dice += dice_coeff3D(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def multiclass_dice_coeff3D(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):#分不同的channel来进行计算
        dice += dice_coeff3D(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]
def dice_loss3D(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff3D if multiclass else dice_coeff3D
    return 1 - fn(input, target, reduce_batch_first=True)
def false_negative_rate_g3D(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###假阴性概率！！！ 用于专门计算绿色的一个
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)
        num=1
        score=(sets_sum-inter + epsilon) / (sets_sum + epsilon)
        if sets_sum==0:#此时完全没有绿色样本
            num=0
            score=torch.zeros(1)
            score=score.cuda()
        # if sets_sum.item() == 0:
        #     sets_sum = 2 * inter
        return score,num
    else:
        # compute and average metric for each batch element
        rate = torch.zeros(1)
        rate=rate.cuda()
        total_num=0
        for i in range(input.shape[0]):
            score,num=false_negative_rate_g3D(input[i, ...], target[i, ...])
            rate+=score
            total_num+=num

        if total_num:
            return rate / total_num, total_num
        else:
            return rate, total_num
def dice_coeff_g3D(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        num=1
        score=(2 * inter + epsilon) / (sets_sum + epsilon)
        if torch.sum(target).item() == 0:#
            num = 0
            score = torch.zeros(1)
            score=score.cuda()

        return score,num
    else:
        # compute and average metric for each batch element
        rate = torch.zeros(1)
        rate=rate.cuda()
        total_num = 0
        for i in range(input.shape[0]):
            score, num = dice_coeff_g3D(input[i, ...], target[i, ...])
            rate += score
            total_num += num
        if total_num:
            return rate / total_num, total_num
        else:
            return rate,total_num
def false_positive_rate3D(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###对于二分类而言 假阳性概率
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)
        pred_sum = torch.sum(input)
        total_num=1
        for i in range(input.dim()):
            total_num*=input.shape[i]
        # if sets_sum.item() == 0:
        #     sets_sum = 2 * inter
        return (pred_sum-inter + epsilon) / (total_num-sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += false_positive_rate3D(input[i, ...], target[i, ...])
        return rate / input.shape[0]
def false_negative_rate3D(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###对于二分类而言 假阴性概率
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)

        # if sets_sum.item() == 0:
        #     sets_sum = 2 * inter
        return (sets_sum-inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += false_negative_rate3D(input[i, ...], target[i, ...])
        return rate / input.shape[0]





#muti-class
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
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
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):#分不同的channel来进行计算
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
def false_negative_rate(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###假阴性概率！！！其实就是binary状态
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)
        if sets_sum.item() == 0:#若此时没有前景，即ROI中不含有绿色区域
            score = torch.zeros(1)
            score=score.cuda()
            return score
        return (sets_sum-inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += false_negative_rate(input[i, ...], target[i, ...])
        return rate / input.shape[0]
def false_positive_rate(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###对于二分类而言 假阳性概率
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)
        pred_sum = torch.sum(input)
        total_num=1
        for i in range(input.dim()):
            total_num*=input.shape[i]
        return (pred_sum-inter + epsilon) / (total_num-sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += false_positive_rate(input[i, ...], target[i, ...])
        return rate / input.shape[0]
def dice_coeff_g(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        num=1
        score=(2 * inter + epsilon) / (sets_sum + epsilon)
        if torch.sum(target).item() == 0:#
            num = 0
            score = torch.zeros(1)
            score=score.cuda()

        return score,num
    else:
        # compute and average metric for each batch element
        rate = torch.zeros(1)
        rate=rate.cuda()
        total_num = 0
        for i in range(input.shape[0]):
            score, num = dice_coeff_g(input[i, ...], target[i, ...])
            rate += score
            total_num += num
        if total_num:
            return rate / total_num, total_num
        else:
            return rate,total_num
#binary
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
def B_dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn =  B_dice_coeff
    return 1 - fn(input[:,0,...], target[:,0,...], reduce_batch_first=True)
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
def B_false_negative_rate3D(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###假阴性概率！！！
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)

        # if sets_sum.item() == 0:
        #     sets_sum = 2 * inter
        return (sets_sum-inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += false_negative_rate3D(input[i, ...], target[i, ...])
        return rate / input.shape[0]
def B_false_positive_rate3D(input: Tensor, target: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###对于二分类而言 假阳性概率
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 3 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(target)
        pred_sum = torch.sum(input)
        total_num=1
        for i in range(input.dim()):
            total_num*=input.shape[i]


        # if sets_sum.item() == 0:
        #     sets_sum = 2 * inter
        return (pred_sum-inter + epsilon) / (total_num-sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += false_positive_rate3D(input[i, ...], target[i, ...])
        return rate / input.shape[0]
def B_recall_g_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # 此处的target是绿色one-hot标签 ,2D求法
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

def B_precision_g_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff

        inter = torch.dot(input.reshape(-1), target.reshape(-1))  # 取交集
        sets_sum = torch.sum(input)
        return (inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += B_precision_g_coeff(input[i, ...], target[i, ...])
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



#masked loss
def masked_dice_coeff(input: Tensor, target: Tensor, ROI: Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    #roi.dim=3
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        input=torch.mul(input,ROI)#无效区域全部乘以0就完事了
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        #inter=torch.dot(inter.reshape(-1), ROI.reshape(-1))#确立感兴趣区域
        sets_sum = torch.sum(torch.dot(input.reshape(-1), ROI.reshape(-1))) + torch.sum(torch.dot(target.reshape(-1), ROI.reshape(-1)))
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
        dice += masked_dice_coeff(input[:, channel, ...], target[:, channel, ...], ROI[:, channel, ...],reduce_batch_first, epsilon)
    return dice / input.shape[1]
def masked_dice_loss(input: Tensor, target: Tensor, ROI:Tensor,multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = masked_multiclass_dice_coeff if multiclass else masked_dice_coeff
    return 1 - fn(input, target,ROI, reduce_batch_first=True)
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
def masked_B_false_negative_rate(input: Tensor, target: Tensor,ROI:Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###假阴性概率！！！
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        input = torch.mul(input, ROI)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(torch.dot(target.reshape(-1), ROI.reshape(-1)))
        if sets_sum.item() == 0:#若此时没有前景，即ROI中不含有绿色区域
            score = torch.zeros(1)
            score=score.cuda()
            return score
        return (sets_sum-inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += masked_B_false_negative_rate(input[i, ...], target[i, ...],ROI[i,...])
        return rate / input.shape[0]
def masked_B_false_positive_rate(input: Tensor, target: Tensor,ROI:Tensor,reduce_batch_first: bool = False, epsilon=1e-6):
    ###对于二分类而言 假阳性概率
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #直接计算一整个batch的Dice coeff
        input = torch.mul(input, ROI)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))#取交集
        sets_sum = torch.sum(torch.dot(target.reshape(-1), ROI.reshape(-1)))
        pred_sum = torch.sum(torch.dot(input.reshape(-1), ROI.reshape(-1)))
        total_num=torch.sum(ROI)
        # if sets_sum.item() == 0:
        #     sets_sum = 2 * inter
        return (pred_sum-inter + epsilon) / (total_num-sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        rate = 0
        for i in range(input.shape[0]):
            rate += masked_B_false_positive_rate(input[i, ...], target[i, ...],ROI[i, ...])
        return rate / input.shape[0]

class masked_CE_loss(nn.Module):
    def __init__(self,  size_average=True,delta=None):
        super(masked_CE_loss, self).__init__()
        self.size_average = size_average
        self.delta = delta
    def forward(self, input, target,ROI):#要求target不是one-hot

        # input=torch.clamp(input,1e-4,1-1e-4)
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
        if self.delta==None:
            delta_list=None
        else:
            delta_list = self.delta
            delta_list = torch.Tensor(delta_list)

        logpt = torch.softmax(input, dim=1)  # N*H*W,C 先进行softmax 再进行log操作
        logpt = torch.clamp(logpt, 1e-4, 1 - 1e-4)
        logpt = torch.log(logpt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        if delta_list is not None:#此时代表有权重
            if delta_list.type() != input.data.type():
                delta_list = delta_list.type_as(input.data)
            at = delta_list.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1*logpt
        # 牛的牛的牛的

        if self.size_average:
            return loss.mean()
        else:
            print('False')



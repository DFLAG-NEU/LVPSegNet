import torch
from torch import nn
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork
class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1).fill_(1.0))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1).fill_(0.0))

    def forward(self, x):
        x = self.a * x + self.b * self.bn(x)
        return x
class Generator1_CAN8(nn.Module):
    def __init__(self,class_num=1):
        super(Generator1_CAN8, self).__init__()
        chn = 64
        self.class_num=class_num
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        # 空洞卷积计算公式: [x+2p-k-(k-1)*(d-1)]/s + 1,中括号表示向下取整
        self.g1_conv1 = nn.Conv2d(1, chn, 3, dilation=1, padding=1)
        self.g1_conv2 = nn.Conv2d(chn, chn, 3, dilation=1, padding=1)
        self.g1_conv3 = nn.Conv2d(chn, chn * 2, 3, dilation=2, padding=2)
        self.g1_conv4 = nn.Conv2d(chn * 2, chn * 4, 3, dilation=4, padding=4)
        self.g1_conv5 = nn.Conv2d(chn * 4, chn * 8, 3, dilation=8, padding=8)
        self.g1_conv6 = nn.Conv2d(chn * 8, chn * 4, 3, dilation=4, padding=4)
        self.g1_conv7 = nn.Conv2d(chn * 4, chn * 2, 3, dilation=2, padding=2)
        self.g1_conv8 = nn.Conv2d(chn * 2, chn, 3, dilation=1, padding=1)
        self.g1_conv9 = nn.Conv2d(chn, class_num, 1, dilation=1)

        self.g1_bn1 = nn.BatchNorm2d(chn)
        self.g1_bn2 = nn.BatchNorm2d(chn)
        self.g1_bn3 = nn.BatchNorm2d(chn * 2)
        self.g1_bn4 = nn.BatchNorm2d(chn * 4)
        self.g1_bn5 = nn.BatchNorm2d(chn * 8)
        self.g1_bn6 = nn.BatchNorm2d(chn * 4)
        self.g1_bn7 = nn.BatchNorm2d(chn * 2)
        self.g1_bn8 = nn.BatchNorm2d(chn)

    def forward(self, input_images):  # 输入[B, 1, 128, 128],输出[B, 1, 128, 128]
        net = self.g1_conv1(input_images)
        net = self.g1_bn1(net)
        net = self.leakyrelu1(net)

        net = self.g1_conv2(net)
        net = self.g1_bn2(net)
        net = self.leakyrelu2(net)

        net = self.g1_conv3(net)
        net = self.g1_bn3(net)
        net = self.leakyrelu3(net)

        net = self.g1_conv4(net)
        net = self.g1_bn4(net)
        net = self.leakyrelu4(net)

        net = self.g1_conv5(net)
        net = self.g1_bn5(net)
        net = self.leakyrelu5(net)

        net = self.g1_conv6(net)
        net = self.g1_bn6(net)
        net = self.leakyrelu6(net)

        net = self.g1_conv7(net)
        net = self.g1_bn7(net)
        net = self.leakyrelu7(net)

        net = self.g1_conv8(net)
        net = self.g1_bn8(net)
        net = self.leakyrelu8(net)

        output = self.g1_conv9(net)

        return output


class Generator2_UCAN64(nn.Module):
    def __init__(self,class_num=1):
        super(Generator2_UCAN64, self).__init__()
        chn = 64
        self.class_num=class_num
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        self.leakyrelu9 = nn.LeakyReLU(0.2)
        self.leakyrelu10 = nn.LeakyReLU(0.2)
        self.leakyrelu11 = nn.LeakyReLU(0.2)
        self.leakyrelu12 = nn.LeakyReLU(0.2)
        self.leakyrelu13 = nn.LeakyReLU(0.2)

        self.g2_conv1 = nn.Conv2d(1, chn, 3, dilation=1, padding=1)
        self.g2_conv2 = nn.Conv2d(chn, chn, 3, dilation=2, padding=2)
        self.g2_conv3 = nn.Conv2d(chn, chn, 3, dilation=4, padding=4)
        self.g2_conv4 = nn.Conv2d(chn, chn, 3, dilation=8, padding=8)
        self.g2_conv5 = nn.Conv2d(chn, chn, 3, dilation=16, padding=16)
        self.g2_conv6 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv7 = nn.Conv2d(chn, chn, 3, dilation=64, padding=64)
        self.g2_conv8 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv9 = nn.Conv2d(chn * 2, chn, 3, dilation=16, padding=16)
        self.g2_conv10 = nn.Conv2d(chn * 2, chn, 3, dilation=8, padding=8)
        self.g2_conv11 = nn.Conv2d(chn * 2, chn, 3, dilation=4, padding=4)
        self.g2_conv12 = nn.Conv2d(chn * 2, chn, 3, dilation=2, padding=2)
        self.g2_conv13 = nn.Conv2d(chn * 2, chn, 3, dilation=1, padding=1)
        self.g2_conv14 = nn.Conv2d(chn, class_num, 1, dilation=1)

        self.g2_bn1 = nn.BatchNorm2d(chn)
        self.g2_bn2 = nn.BatchNorm2d(chn)
        self.g2_bn3 = nn.BatchNorm2d(chn)
        self.g2_bn4 = nn.BatchNorm2d(chn)
        self.g2_bn5 = nn.BatchNorm2d(chn)
        self.g2_bn6 = nn.BatchNorm2d(chn)
        self.g2_bn7 = nn.BatchNorm2d(chn)
        self.g2_bn8 = nn.BatchNorm2d(chn)
        self.g2_bn9 = nn.BatchNorm2d(chn)
        self.g2_bn10 = nn.BatchNorm2d(chn)
        self.g2_bn11 = nn.BatchNorm2d(chn)
        self.g2_bn12 = nn.BatchNorm2d(chn)
        self.g2_bn13 = nn.BatchNorm2d(chn)

    def forward(self, input_images):  # 输入[B, 1, 128, 128],输出[B, 1, 128, 128]
        net1 = self.g2_conv1(input_images)
        net1 = self.g2_bn1(net1)
        net1 = self.leakyrelu1(net1)

        net2 = self.g2_conv2(net1)
        net2 = self.g2_bn2(net2)
        net2 = self.leakyrelu2(net2)

        net3 = self.g2_conv3(net2)
        net3 = self.g2_bn3(net3)
        net3 = self.leakyrelu3(net3)

        net4 = self.g2_conv4(net3)
        net4 = self.g2_bn4(net4)
        net4 = self.leakyrelu4(net4)

        net5 = self.g2_conv5(net4)
        net5 = self.g2_bn5(net5)
        net5 = self.leakyrelu5(net5)

        net6 = self.g2_conv6(net5)
        net6 = self.g2_bn6(net6)
        net6 = self.leakyrelu6(net6)

        net7 = self.g2_conv7(net6)
        net7 = self.g2_bn7(net7)
        net7 = self.leakyrelu7(net7)

        net8 = self.g2_conv8(net7)
        net8 = self.g2_bn8(net8)
        net8 = self.leakyrelu8(net8)

        net9 = torch.cat([net6, net8], dim=1)

        net9 = self.g2_conv9(net9)
        net9 = self.g2_bn9(net9)
        net9 = self.leakyrelu9(net9)

        net10 = torch.cat([net5, net9], dim=1)

        net10 = self.g2_conv10(net10)
        net10 = self.g2_bn10(net10)
        net10 = self.leakyrelu10(net10)

        net11 = torch.cat([net4, net10], dim=1)

        net11 = self.g2_conv11(net11)
        net11 = self.g2_bn11(net11)
        net11 = self.leakyrelu11(net11)

        net12 = torch.cat([net3, net11], dim=1)

        net12 = self.g2_conv12(net12)
        net12 = self.g2_bn12(net12)
        net12 = self.leakyrelu12(net12)

        net13 = torch.cat([net2, net12], dim=1)

        net13 = self.g2_conv13(net13)
        net13 = self.g2_bn13(net13)
        net13 = self.leakyrelu13(net13)

        net14 = self.g2_conv14(net13)

        return net14


class seg_net(nn.Module):
    def __init__(self,class_num=1):
        super(seg_net, self).__init__()
        self.moduleFN=Generator1_CAN8(class_num)
        self.moduleFP=Generator2_UCAN64(class_num)
        #两个网络相结合
    def forward(self, input_images):  # 输入[B, 1, 128, 128],输出[B, 1, 128, 128]
        output_FN=self.moduleFN(input_images)
        output_FP = self.moduleFP(input_images)

        # output_FN = F.softmax(output_FN, dim=1)
        # output_FP = F.softmax(output_FP, dim=1)

        #必须要进行一次softmax,是否有必要？

        output = (output_FN + output_FP) / 2


        return output

class final_seg_net(SegmentationNetwork):
    def __init__(self,class_num=1):
        super(final_seg_net, self).__init__()
        self.params={'content':None}
        self.conv_op=nn.Conv2d
        self.do_ds=True
        self.num_classes=class_num

        self.model=seg_net(class_num)

        #两个网络相结合
    def forward(self, x):  # 输入[B, 1, 128, 128],输出[B, 1, 128, 128]
        if self.do_ds:
            return[self.model(x),]
        else:
            return self.model(x)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
if __name__=="__main__":
    a=seg_net()
    a=Generator2_UCAN64()
    b=Generator1_CAN8()
    # print(get_parameter_number(a)['Total'])
    # print(get_parameter_number(b)['Total'])
    # print(get_parameter_number(c)['Total'])
    # print(get_parameter_number(a)['Total']+get_parameter_number(b)['Total']+get_parameter_number(c)['Total'])

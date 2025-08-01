import torch
from torch import nn
import torch.nn.functional as F
import sys

class kiSTSN_plus(nn.Module):

    def __init__(self,class_num=4,softmax_flag=False,fusion_flag=True,direction='L2H',fusion_mode=0):
        super(kiSTSN_plus, self).__init__()
        class_num=class_num
        self.softmax_flag=softmax_flag
        self.fusion_flag=fusion_flag
        chn = 64
        self.class_num=class_num
        self.g1_leakyrelu1 = nn.LeakyReLU(0.2)
        self.g1_leakyrelu2 = nn.LeakyReLU(0.2)
        self.g1_leakyrelu3 = nn.LeakyReLU(0.2)
        self.g1_leakyrelu4 = nn.LeakyReLU(0.2)
        self.g1_leakyrelu5 = nn.LeakyReLU(0.2)
        self.g1_leakyrelu6 = nn.LeakyReLU(0.2)
        self.g1_leakyrelu7 = nn.LeakyReLU(0.2)
        self.g1_leakyrelu8 = nn.LeakyReLU(0.2)
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


#######################
        self.g2_leakyrelu1 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu2 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu3 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu4 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu5 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu6 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu7 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu8 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu9 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu10 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu11 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu12 = nn.LeakyReLU(0.2)
        self.g2_leakyrelu13 = nn.LeakyReLU(0.2)

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

        ################编码器区域

#1 分支代表 low to high
#2 分支代表 high to low
        self.direction=direction
        self.fusion_mode=fusion_mode
        if self.fusion_flag:

            self.inter1_1 = nn.Conv2d(64+128, 128, 1)
            self.inter1_1bn = nn.BatchNorm2d(128)
            self.inter2_1 = nn.Conv2d(64+512, 512, 1)
            self.inter2_1bn = nn.BatchNorm2d(512)
            self.inter3_1 = nn.Conv2d(64+128, 128, 1)
            self.inter3_1bn = nn.BatchNorm2d(128)

            self.inter1_2 = nn.Conv2d(128 + 64, 64, 1)
            self.inter1_2bn = nn.BatchNorm2d(64)
            self.inter2_2 = nn.Conv2d(512 + 64, 64, 1)
            self.inter2_2bn = nn.BatchNorm2d(64)
            self.inter3_2 = nn.Conv2d(128 + 64, 64, 1)
            self.inter3_2bn = nn.BatchNorm2d(64)

            self.inter1_1_plus = nn.Conv2d(64+256, 256, 1)
            self.inter1_1bn_plus = nn.BatchNorm2d(256)
            self.inter2_1_plus = nn.Conv2d(64+256, 256, 1)
            self.inter2_1bn_plus = nn.BatchNorm2d(256)

            self.inter1_2_plus = nn.Conv2d(256 + 64, 64, 1)
            self.inter1_2bn_plus = nn.BatchNorm2d(64)
            self.inter2_2_plus = nn.Conv2d(256 + 64, 64, 1)
            self.inter2_2bn_plus = nn.BatchNorm2d(64)

        if self.fusion_mode==1:
            #类似kiunet的形式
            self.final_conv = nn.Conv2d(chn, class_num, 1, dilation=1)

    def forward(self, x):
        input_images=x

        ######################G1_S1#############################
        g1_out = self.g1_conv1(input_images)
        g1_out = self.g1_bn1(g1_out)
        g1_out = self.g1_leakyrelu1(g1_out)

        g1_out = self.g1_conv2(g1_out)
        g1_out = self.g1_bn2(g1_out)
        g1_out = self.g1_leakyrelu2(g1_out)
        #torch.Size([2, 64, 64, 64])
        g1_out = self.g1_conv3(g1_out)
        g1_out = self.g1_bn3(g1_out)
        g1_out = self.g1_leakyrelu3(g1_out)
        #torch.Size([2, 128, 64, 64])
        #####################G2_S1##############################

        g2_net1 = self.g2_conv1(input_images)
        g2_net1 = self.g2_bn1(g2_net1)
        g2_net1 = self.g2_leakyrelu1(g2_net1)#torch.Size([2, 64, 64, 64])

        g2_net2 = self.g2_conv2(g2_net1)
        g2_net2 = self.g2_bn2(g2_net2)
        g2_net2 = self.g2_leakyrelu2(g2_net2)

        g2_net3 = self.g2_conv3(g2_net2)
        g2_net3 = self.g2_bn3(g2_net3)
        g2_net3 = self.g2_leakyrelu3(g2_net3)

        if self.fusion_flag:
            tmp=torch.cat((g1_out,g2_net3),dim=1)
            if self.direction=='L2H':
                g2_net3 = F.relu(self.inter1_2bn(self.inter1_2(tmp)))
            else:
                sys.exit()




        #即将在此处发生交换
        g1_out = self.g1_conv4(g1_out)
        g1_out = self.g1_bn4(g1_out)
        g1_out = self.g1_leakyrelu4(g1_out)

        g2_net4 = self.g2_conv4(g2_net3)
        g2_net4 = self.g2_bn4(g2_net4)
        g2_net4 = self.g2_leakyrelu4(g2_net4)

        g2_net5 = self.g2_conv5(g2_net4)
        g2_net5 = self.g2_bn5(g2_net5)
        g2_net5 = self.g2_leakyrelu5(g2_net5)

        #此处附加交换
        #此处附加交换
        #此处附加交换
        if self.fusion_flag:

            tmp=torch.cat((g1_out,g2_net5),dim=1)

            if self.direction=='L2H':
                g2_net5 = F.relu(self.inter1_2bn_plus(self.inter1_2_plus(tmp)))
            else:
                sys.exit()

        #torch.Size([2, 256, 64, 64])
        g1_out = self.g1_conv5(g1_out)
        g1_out = self.g1_bn5(g1_out)
        g1_out = self.g1_leakyrelu5(g1_out)
        #torch.Size([2, 512, 64, 64])




        g2_net6 = self.g2_conv6(g2_net5)
        g2_net6 = self.g2_bn6(g2_net6)
        g2_net6 = self.g2_leakyrelu6(g2_net6)

        g2_net7 = self.g2_conv7(g2_net6)
        g2_net7 = self.g2_bn7(g2_net7)
        g2_net7 = self.g2_leakyrelu7(g2_net7)

        if self.fusion_flag:
            tmp=torch.cat((g1_out,g2_net7),dim=1)


            if self.direction=='L2H':
                g2_net7 = F.relu(self.inter2_2bn(self.inter2_2(tmp)))  # CRFB
            else:
                sys.exit()



        #即将在此处发生交换
        g1_out = self.g1_conv6(g1_out)
        g1_out = self.g1_bn6(g1_out)
        g1_out = self.g1_leakyrelu6(g1_out)


        g2_net8 = self.g2_conv8(g2_net7)
        g2_net8 = self.g2_bn8(g2_net8)
        g2_net8 = self.g2_leakyrelu8(g2_net8)

        g2_net9 = torch.cat([g2_net6, g2_net8], dim=1)

        g2_net9 = self.g2_conv9(g2_net9)
        g2_net9 = self.g2_bn9(g2_net9)
        g2_net9 = self.g2_leakyrelu9(g2_net9)


        #此处附加交换
        #此处附加交换
        #此处附加交换
        if self.fusion_flag:

            tmp=torch.cat((g1_out,g2_net9),dim=1)
            if self.direction=='L2H':
                g2_net9 = F.relu(self.inter2_2bn_plus(self.inter2_2_plus(tmp)))  # CRFB
            else:
                sys.exit()



        # torch.Size([2, 256, 64, 64])
        g1_out = self.g1_conv7(g1_out)
        g1_out = self.g1_bn7(g1_out)
        g1_out = self.g1_leakyrelu7(g1_out)
        # torch.Size([2, 128, 64, 64])

        g2_net10 = torch.cat([g2_net5, g2_net9], dim=1)

        g2_net10 = self.g2_conv10(g2_net10)
        g2_net10 = self.g2_bn10(g2_net10)
        g2_net10 = self.g2_leakyrelu10(g2_net10)

        g2_net11 = torch.cat([g2_net4, g2_net10], dim=1)

        g2_net11 = self.g2_conv11(g2_net11)
        g2_net11 = self.g2_bn11(g2_net11)
        g2_net11 = self.g2_leakyrelu11(g2_net11)

        if self.fusion_flag:

            tmp=torch.cat((g1_out,g2_net11),dim=1)
            if self.direction=='L2H':
                g2_net11 = F.relu(self.inter3_2bn(self.inter3_2(tmp)))  # CRFB
            else:
                sys.exit()


        #即将在此处发生交换
        g1_out = self.g1_conv8(g1_out)
        g1_out = self.g1_bn8(g1_out)
        g1_out = self.g1_leakyrelu8(g1_out)
        # torch.Size([2, 64, 64, 64])



        g2_net12 = torch.cat([g2_net3, g2_net11], dim=1)

        g2_net12 = self.g2_conv12(g2_net12)
        g2_net12 = self.g2_bn12(g2_net12)
        g2_net12 = self.g2_leakyrelu12(g2_net12)

        g2_net13 = torch.cat([g2_net2, g2_net12], dim=1)

        g2_net13 = self.g2_conv13(g2_net13)
        g2_net13 = self.g2_bn13(g2_net13)
        g2_net13 = self.g2_leakyrelu13(g2_net13)


        if self.fusion_mode==0:
            output_FN = self.g1_conv9(g1_out)
            output_FP = self.g2_conv14(g2_net13)

            output_FN = F.softmax(output_FN, dim=1)
            output_FP = F.softmax(output_FP, dim=1)
            output = (output_FN + output_FP) / 2
            assert self.softmax_flag==True
        elif self.fusion_mode == 1:
            output=self.final_conv(g1_out+g2_net13)
            if self.softmax_flag:
                output=F.softmax(output,dim=1)

        elif self.fusion_mode == 2:
            output_FN = self.g1_conv9(g1_out)
            output_FP = self.g2_conv14(g2_net13)

            output = self.final_conv(torch.cat([output_FN, output_FP], dim=1))
            if self.softmax_flag:
                output=F.softmax(output,dim=1)

        return output


if __name__=="__main__":
    #统计模型容量


    b=torch.zeros((2,1,64,64))
    # d=kiSTSN()
    e=kiSTSN_plus(fusion_mode=2)


    out=e(b)
    print(out.shape)
    print('')

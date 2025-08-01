import logging
from os import listdir
from os.path import splitext
import os
import numpy as np
import torch
from skimage.morphology import disk
from skimage.morphology.binary import binary_erosion,binary_dilation
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms import RandomAffine,Compose
import random
from Y24M7_Restart.Utils_imple.Image_process import norm,normalization,cal_centerofmass,cal_cropbox
#####2D
import json

VOXEL_MAP_DICT={1:'G:/Task_M7D27/Utils/voxel_size_new.json',
                2:'G:/Task_M7D27/Utils/voxel_size_OC.json',
                3:'G:/Task_M7D27/external_RJ/Data/MACE/voxel_info.json',
                4:'G:/Task_M7D27/external_RJ/Data/non-MACE/voxel_info.json',
                5:'G:/Task_M7D27/Utils/voxel_size_EMIDEC.json'}

#能否把CSMRE整合进去
class BasicDataset_2D(Dataset):
    def __init__(self, images_dir, labels_dir,rois_dir='',index_flag=False,Rshift_dis=5,crop_flag=False,roi_aug_flag=False,CL_flag=False,cropsize=128,aug_CL=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.size=100
        self.Rshift_dis=Rshift_dis
        self.roi_flag=False
        self.rois_dir=rois_dir
        if not rois_dir=='':
            self.roi_flag = True
        self.op=(binary_dilation,binary_erosion)
        self.rate=0.5
        self.atlas_list=[]
        self.index_flag = index_flag
        self.atlas_list=os.listdir(images_dir)
        self.crop_flag=crop_flag
        self.cropsize=cropsize
        self.aug_flag=roi_aug_flag
        self.CL_flag=CL_flag

    def __len__(self):
        return len(self.atlas_list)
    def __getitem__(self, idx):
        #

        # print(idx)
        if not self.CL_flag:
            name = self.atlas_list[idx]
        else:
            name=str(idx)+'.npy'

        index = idx

        label = np.load(self.labels_dir+'/'+name)
        label=np.where(label==4,0,label)
        img = np.load(self.images_dir+'/'+name)

        if self.roi_flag:
            roi = np.load(self.rois_dir + '/' + name)
            midX, midY = cal_centerofmass(roi.copy())
        else:
            #直接中心点
            midX=int(img.shape[0]/2)
            midY=int(img.shape[1]/2)


        if self.crop_flag:

            startX, startY = cal_cropbox(midX, midY, img.shape[0], self.cropsize)
            img = img[startX:startX + self.cropsize, startY:startY + self.cropsize]
            label = label[startX:startX + self.cropsize, startY:startY + self.cropsize]
            if self.roi_flag:
                roi = roi[startX:startX + self.cropsize, startY:startY + self.cropsize]
            img = norm(img)

        assert img.size == label.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        a = torch.from_numpy(img).float()
        b = torch.from_numpy(label).long()


        if self.roi_flag:
            if self.aug_flag:
                rd=random.uniform(0,1)
                if rd<self.rate:
                    op=np.random.choice(self.op)
                    selem=disk(random.uniform(1,2),dtype=bool)
                    roi=op(roi.astype(bool),selem)

            # 如果有必要的话，可以输出一下平移后的结果
            c = torch.from_numpy(roi).long()
            if self.aug_flag:
                c = c.unsqueeze(0)
                c = c.unsqueeze(0)
                ##对ROI随机进行平移,并将ROI与图片相乘
                shift_x = random.randint(-self.Rshift_dis, self.Rshift_dis) / self.size
                shift_y = random.randint(-self.Rshift_dis, self.Rshift_dis) / self.size
                transform_matrix = torch.tensor([
                    [1, 0, shift_x],
                    [0, 1, shift_y]]).unsqueeze(0)  # 设B(batch size为1)
                grid = F.affine_grid(transform_matrix,  # 旋转变换矩阵
                                     c.shape)  # 变换后的tensor的shape(与输入tensor相同)

                output = F.grid_sample(c.float(),  # 输入tensor，shape为[B,C,W,H]
                                       grid,  # 上一步输出的gird,shape为[B,C,W,H]
                                       mode='bilinear')  # 一些图像填充方法，这里我用的是最近邻
                c = (output > 0.5).squeeze().long()
            b = b * c

            #################如果有必要进行验证的话，输出平移前后的图像样式
            #################
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]), c.view(
                    roi.shape[0], roi.shape[1]), index
            else:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]), c.view(roi.shape[0],
                                                                                                             roi.shape[1])
        else:
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]), index
            else:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1])




class BasicDataset_3D(Dataset):
    def __init__(self, images_dir, labels_dir,rois_dir='',flag=False,voxel_flag=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.roi_flag=False
        self.index_flag=flag
        if not rois_dir=='':
            self.rois_dir=rois_dir
            self.roi_flag = True
        tmp=os.listdir(images_dir)
        self.atlas_list=tmp
        self.voxel_flag=voxel_flag
        if self.voxel_flag:
            with open(VOXEL_MAP_DICT[self.voxel_flag], "r", encoding="utf-8") as f:
                content_all = json.load(f)
            self.voxel_size_list = content_all
    def __len__(self):
        return len(self.atlas_list)
    def __getitem__(self, idx):
        name = self.atlas_list[idx]
        label = np.load(self.labels_dir+'/'+name)
        img = np.load(self.images_dir+'/'+name)
        assert img.size == label.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        a = torch.from_numpy(img).float()
        b = torch.from_numpy(label).long()

        if self.voxel_flag:#此时需要读取体素信息

            p_index = name[:-4]
            voxel_size=self.voxel_size_list[p_index]
            voxel_size=np.array(voxel_size)
            voxel_size=torch.from_numpy(voxel_size).float()

        if self.roi_flag:
            roi = np.load(self.rois_dir + '/' + name)
            c = torch.from_numpy(roi)
            c = c.long()
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1],img.shape[2]), b.view(label.shape[0], label.shape[1],label.shape[2]), c.view(
                    roi.shape[0], roi.shape[1],roi.shape[2]), idx
            else:
                if self.voxel_flag:
                    return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                       label.shape[2]), c.view(
                        roi.shape[0], roi.shape[1], roi.shape[2]),voxel_size
                else:
                    return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                       label.shape[2]), c.view(
                        roi.shape[0], roi.shape[1], roi.shape[2])

        else:
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                   label.shape[2]), idx
            else:
                if not self.voxel_flag:
                    return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                   label.shape[2])
                else:
                    return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                   label.shape[2]),voxel_size


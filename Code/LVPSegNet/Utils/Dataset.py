import logging
from os import listdir
from os.path import splitext
import os
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from torchvision.transforms import RandomAffine,Compose
from skimage.morphology.binary import binary_erosion,binary_dilation
import random
from skimage.morphology import disk
def cal_center_of_mass(input,method=1):
    height,width=input.shape
    sumx=0
    sumy=0
    area=0
    if method==1:
        posX,posY=np.where(input>0)
        return int((np.max(posX)+np.min(posX))/2),int((np.max(posY)+np.min(posY))/2)
    else:
        for i in range(height):
            for j in range(width):
                if input[i,j]==1:
                    sumx=sumx+i
                    sumy = sumy + j
                    area=area+1
        return int(sumx/area),int(sumy/area)
def norm(input):
    if type(input) is np.ndarray:
        input=(input-np.min(input))/(np.max(input)-np.min(input))
    else:
        input=(input-torch.min(input))/(torch.max(input)-torch.min(input))
    return input
def normalization(input):
    return (input-np.min(input))/(np.max(input)-np.min(input))
def z_score_normalization(input):
    return (input-np.mean(input))/np.std(input)
def interpolate(input,scale,type=1):
    #####type1--image type2--label
    if type==1:
        return zoom(input, (scale, scale), order=3)
    elif type==2:
        return zoom(input, (scale, scale), order=0)
    else:
        print('False')

def cal_cropbox(midX,midY,size,cropsize):
    if isinstance(size,tuple):
        assert cropsize <= size[0]
        assert cropsize <= size[1]
        startX = midX - int(cropsize / 2)
        startY = midY - int(cropsize / 2)
        if midX - int(cropsize / 2) < 0:
            startX = 0
        if midY - int(cropsize / 2) < 0:
            startY = 0
        if midX + int(cropsize / 2) > size[0]:
            startX = size[0] - cropsize
        if midY + int(cropsize / 2) > size[1]:
            startY = size[1] - cropsize
        return startX, startY

    else:
        assert cropsize < size
        startX = midX - int(cropsize / 2)
        startY = midY - int(cropsize / 2)
        if midX - int(cropsize / 2) < 0:
            startX = 0
        if midY - int(cropsize / 2) < 0:
            startY = 0
        if midX + int(cropsize / 2) > size:
            startX = size - cropsize
        if midY + int(cropsize / 2) > size:
            startY = size - cropsize
        return startX, startY


class DirectSeg_Basic_cropped_Dataset(Dataset):#经过定位之后的输入图像
    def __init__(self, images_dir: str, masks_dir: str,flag=False,crop_size=256,normalization_type=0,mode='',class_id=-1,dataindex_list=[]):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.atlas_list=os.listdir(images_dir)
        self.flag=flag
        self.norm_flag=True
        self.cropsize=crop_size
        self.normalization_type=normalization_type
        self.class_id=class_id
        self.mode=mode

        tmp=os.listdir(images_dir)
        self.atlas_list=[]
        if not len(dataindex_list)==0:
           for i in dataindex_list:
               (self.atlas_list).append(str(i)+'.npy')
        else:
            self.atlas_list=tmp

    def __len__(self):
        return len(self.atlas_list)

    def __getitem__(self, idx):
        name = self.atlas_list[idx]
        mask = np.load(self.masks_dir+'/'+name)
        img = np.load(self.images_dir+'/'+name)
        midX=int(img.shape[0]/2)
        midY=int(img.shape[0]/2)
        startX,startY=cal_cropbox(midX,midY,img.shape[0],self.cropsize)
        mask=mask[startX:startX+self.cropsize,startY:startY+self.cropsize]
        img=img[startX:startX+self.cropsize,startY:startY+self.cropsize]
        if self.mode=='Myo':
            mask=np.where(mask>0,1,0)
        if self.mode=='Myo&Cav':
            mask=np.where(mask==3,1,mask)
            mask = np.where(mask == 2, 1, mask)
            mask=np.where(mask==4,2,mask)
        if self.mode=='Ring':
            mask=np.where(mask==4,0,mask)
        if not  self.class_id==-1:
            mask=np.where(mask==self.class_id,1,0)

        if self.norm_flag:
            if not self.normalization_type==1:
                img=normalization(img)
            else:
                img = z_score_normalization(img)
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        a = torch.from_numpy(img)
        a = a.float()
        b = torch.from_numpy(mask)
        b = b.long()
        if self.flag:
            return  a.view(1,self.cropsize,self.cropsize),b.view(self.cropsize,self.cropsize),idx
        else:
            return  a.view(1,self.cropsize,self.cropsize),b.view(self.cropsize,self.cropsize)


class BasicDataset_3D(Dataset):
    def __init__(self, images_dir, labels_dir,rois_dir='',rois_dir2='',flag=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.roi_flag=False
        self.index_flag=flag
        if not rois_dir=='':
            self.rois_dir=rois_dir
            self.roi_flag = True
        self.rois_dir2=''
        if not rois_dir2=='':
            self.rois_dir2=rois_dir2
        tmp=os.listdir(images_dir)
        self.atlas_list=tmp
    def __len__(self):
        return len(self.atlas_list)
    def __getitem__(self, idx):
        name = self.atlas_list[idx]
        label = np.load(self.labels_dir+'/'+name)
        img = np.load(self.images_dir+'/'+name)
        assert img.size == label.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        a = torch.from_numpy(img)
        a = a.float()
        b = torch.from_numpy(label)
        b = b.long()
        if self.roi_flag:
            roi = np.load(self.rois_dir + '/' + name)
            c = torch.from_numpy(roi)
            c = c.long()
            if not self.rois_dir2=='':
                roi_pre = np.load(self.rois_dir2 + '/' + name)
                d = torch.from_numpy(roi_pre)
                d = d.long()
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1],img.shape[2]), b.view(label.shape[0], label.shape[1],label.shape[2]), c.view(
                    roi.shape[0], roi.shape[1],roi.shape[2]), idx
            else:
                if not self.rois_dir2 == '':
                    return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                       label.shape[2]), c.view(
                        roi.shape[0], roi.shape[1], roi.shape[2]), d.view(
                        roi.shape[0], roi.shape[1], roi.shape[2])
                else:
                    return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                       label.shape[2]), c.view(
                        roi.shape[0], roi.shape[1], roi.shape[2])

        else:
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                   label.shape[2]), idx
            else:
                return a.view(1, img.shape[0], img.shape[1], img.shape[2]), b.view(label.shape[0], label.shape[1],
                                                                                   label.shape[2])

class BasicDataset_2D_ROI(Dataset):
    def __init__(self, images_dir, labels_dir, rois_dir, flag=False, crop_flag=False, dataindex_list=[]):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.rois_dir=rois_dir
        self.index_flag = flag
        tmp = os.listdir(images_dir)
        self.atlas_list = []
        if not len(dataindex_list) == 0:
            for i in dataindex_list:
                (self.atlas_list).append(str(i) + '.npy')
        else:
            self.atlas_list = tmp
        self.crop_flag = crop_flag
        self.cropsize=128

    def __len__(self):
        return len(self.atlas_list)

    def __getitem__(self, idx):
        name = self.atlas_list[idx]
        label = np.load(self.labels_dir + '/' + name)

        label = np.where(label == 4, 0, label)

        img = np.load(self.images_dir + '/' + name)
        assert img.size == label.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        a = torch.from_numpy(img)
        a = a.float()
        roi = np.load(self.rois_dir + '/' + name)
        label=label*roi
        if self.crop_flag:
            midX,midY=cal_center_of_mass(roi.copy())
            startX, startY = cal_cropbox(midX, midY, img.shape[0], self.cropsize)
        b = torch.from_numpy(label)
        b = b.long()

        c = torch.from_numpy(roi)
        c = c.long()

        if self.crop_flag:
            ### forget to normalization
            a = a[startX:startX + self.cropsize, startY:startY + self.cropsize]
            b = b[startX:startX + self.cropsize, startY:startY + self.cropsize]
            c = c[startX:startX + self.cropsize, startY:startY + self.cropsize]
            a=norm(a)
        if self.index_flag:
            if self.crop_flag:
                return a.view(1, self.cropsize, self.cropsize), b.view(self.cropsize, self.cropsize), c.view(
                    self.cropsize, self.cropsize), idx
            else:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]), c.view(
                    roi.shape[0], roi.shape[1]), idx
        else:
            if self.crop_flag:
                return a.view(1, self.cropsize, self.cropsize), b.view(self.cropsize, self.cropsize), c.view(
                    self.cropsize, self.cropsize)
            else:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]), c.view(
                    roi.shape[0], roi.shape[1])

class DirectSeg_Basic_cropped_Dataset_noaug(Dataset):#经过定位之后的输入图像
    def __init__(self, images_dir: str, masks_dir: str,flag=False,crop_size=256,normalization_type=0,mode='',class_id=-1,dataindex_list=[]):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.atlas_list=os.listdir(images_dir)
        self.flag=flag
        self.norm_flag=True
        self.cropsize=crop_size
        self.normalization_type=normalization_type
        self.class_id=class_id
        self.mode=mode
        self.show_flag=False
        tmp=os.listdir(images_dir)
        self.atlas_list=[]
        if not len(dataindex_list)==0:
           for i in dataindex_list:
               (self.atlas_list).append(str(i)+'.npy')
        else:
            self.atlas_list=tmp

    def __len__(self):
        return len(self.atlas_list)

    def __getitem__(self, idx):
        name = self.atlas_list[idx]

        mask = np.load(self.masks_dir+'/'+name)
        img = np.load(self.images_dir+'/'+name)

        img = normalization(img)
        #print(img.shape)
        if img.shape[0]<self.cropsize:
            img=np.pad(img,((int((self.cropsize-img.shape[0])/2), (self.cropsize-img.shape[0])- int((self.cropsize-img.shape[0])/2)  ),(0,0)))
        if img.shape[1] < self.cropsize:
            img=np.pad(img,((0,0),(int((self.cropsize-img.shape[1])/2), (self.cropsize-img.shape[1])- int((self.cropsize-img.shape[1])/2)  )))
        if mask.shape[0]<self.cropsize:
            mask=np.pad(mask,((int((self.cropsize-mask.shape[0])/2), (self.cropsize-mask.shape[0])- int((self.cropsize-mask.shape[0])/2)  ),(0,0)))
        if mask.shape[1] < self.cropsize:
            mask=np.pad(mask,((0,0),(int((self.cropsize-mask.shape[1])/2), (self.cropsize-mask.shape[1])- int((self.cropsize-mask.shape[1])/2)  )))


        #print(img.shape)
        midX=int(img.shape[0]/2)
        midY=int(img.shape[1]/2)
        #maybe padding?
        startX,startY=cal_cropbox(midX,midY,(img.shape[0],img.shape[1]),self.cropsize)
        mask=mask[startX:startX+self.cropsize,startY:startY+self.cropsize]
        img=img[startX:startX+self.cropsize,startY:startY+self.cropsize]
        img = normalization(img)

        # if self.show_flag:
        #     before_aug = combine_imglabel(img.copy(), mask)
        #     before_aug.save('train_show/%s.png' % name[:-4])

        a = torch.from_numpy(img)
        a = a.float()
        a=a.view(1,1,self.cropsize,self.cropsize)
        b = torch.from_numpy(mask)
        b = b.long()


        if self.show_flag:
            after_aug = combine_imglabel(((a.numpy())[0, 0]).copy(), (b.numpy())[0])
            after_aug.save('train_show/%s_after.png' % name[:-4])

        if self.flag:
            return  a.view(1,self.cropsize,self.cropsize),b.view(self.cropsize,self.cropsize),idx
        else:
            return  a.view(1,self.cropsize,self.cropsize),b.view(self.cropsize,self.cropsize)
class BasicDataset_2D_ROIshift_DE(Dataset):
    def __init__(self, images_dir, labels_dir,rois_dir='',flag=False,dataindex_list=[],Rshift_dis=5,crop_flag=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.size=100
        self.Rshift_dis=Rshift_dis
        self.roi_flag=False
        self.index_flag=flag
        if not rois_dir=='':
            self.rois_dir=rois_dir
            self.roi_flag = True
        tmp=os.listdir(images_dir)
        self.op=(binary_dilation,binary_erosion)
        self.rate=0.5
        self.atlas_list=[]
        if not len(dataindex_list)==0:
           for i in dataindex_list:
               (self.atlas_list).append(str(i)+'.npy')
        else:
            self.atlas_list=tmp
        self.crop_flag=crop_flag
        self.cropsize=128
    def __len__(self):
        return len(self.atlas_list)
    def __getitem__(self, idx):
        name = self.atlas_list[idx]
        label = np.load(self.labels_dir+'/'+name)
        label=np.where(label==4,0,label)
        img = np.load(self.images_dir+'/'+name)
        roi = np.load(self.rois_dir + '/' + name)
        if self.crop_flag:
            midX, midY = cal_center_of_mass(roi.copy())
            startX, startY = cal_cropbox(midX, midY, img.shape[0], self.cropsize)
            img = img[startX:startX + self.cropsize, startY:startY + self.cropsize]
            label = label[startX:startX + self.cropsize, startY:startY + self.cropsize]
            roi = roi[startX:startX + self.cropsize, startY:startY + self.cropsize]
            img = norm(img)
        assert img.size == label.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        a = torch.from_numpy(img)
        a = a.float()
        b = torch.from_numpy(label)
        b = b.long()
        rd=random.uniform(0,1)
        if rd<self.rate:
            op=np.random.choice(self.op)
            selem=disk(random.uniform(1,2),dtype=bool)
            roi=op(roi.astype(bool),selem)
        if self.roi_flag:
            #如果有必要的话，可以输出一下平移后的结果

            c = torch.from_numpy(roi)
            c = c.long()
            c = c.unsqueeze(0)
            c = c.unsqueeze(0)
            ##对ROI随机进行平移,并将ROI与图片相乘
            shift_x = random.randint(-self.Rshift_dis,self.Rshift_dis)/self.size
            shift_y = random.randint(-self.Rshift_dis,self.Rshift_dis)/self.size
            transform_matrix = torch.tensor([
                [1, 0, shift_x],
                [0, 1, shift_y]]).unsqueeze(0)  # 设B(batch size为1)
            grid = F.affine_grid(transform_matrix,  # 旋转变换矩阵
                                 c.shape)  # 变换后的tensor的shape(与输入tensor相同)
            c=c.float()

            output = F.grid_sample(c,  # 输入tensor，shape为[B,C,W,H]
                                   grid,  # 上一步输出的gird,shape为[B,C,W,H]
                                   mode='bilinear')  # 一些图像填充方法，这里我用的是最近邻
            output=output>0.5
            c=output.squeeze()
            c = c.long()
            b=b*c

            #################如果有必要进行验证的话，输出平移前后的图像样式
            #################
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]), c.view(
                    roi.shape[0], roi.shape[1]), idx
            else:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]),c.view(roi.shape[0], roi.shape[1])
        else:
            if self.index_flag:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1]),idx
            else:
                return a.view(1, img.shape[0], img.shape[1]), b.view(label.shape[0], label.shape[1])
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from scipy.ndimage.interpolation import zoom
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import pydicom
import nibabel as nib

def normalization(input,value_max,value_min):
    input=(input-np.min(input))/(np.max(input)-np.min(input))
    input=input*(value_max-value_min)
    input=input+value_min
    return input
def nii3D_show(readpath,savepath):
    tmp=nib.load(readpath)
    img=tmp.get_fdata()
    X,Y,Z=img.shape
    slice_num=min(min(X,Y),Z)

    for i in range(slice_num):#
        slice_tmp=img[:,:,i].copy()
        slice_tmp=np.swapaxes(slice_tmp,0,1)
        # slice_tmp=np.where(slice_tmp>0,1,0)
        slice_tmp=normalization(slice_tmp,np.max(slice_tmp),np.min(slice_tmp))
        image(slice_tmp,saveflag=True,save_path=savepath+'/%d.png'%i)

def nii3D_label_show(readpath,labelpath,savepath):
    tmp=nib.load(readpath)
    img=tmp.get_fdata()
    lb=nib.load(labelpath)
    label=lb.get_fdata()
    X,Y,Z=img.shape

    assert X==label.shape[0]
    assert Y == label.shape[1]
    assert Z == label.shape[2]
    slice_num=min(min(X,Y),Z)
    for i in range(slice_num):#
        slice_tmp=img[:,:,i].copy()
        slice_label=label[:,:,i].copy()
        # slice_tmp=np.swapaxes(slice_tmp,0,1)
        slice_tmp=normalization(slice_tmp,1,0)
        combine_imglabel(slice_tmp,slice_label,dye_flag=True,save_flag=True,save_dir=savepath+'-%d.png'%i)
def interpolate(input,scale,type=1):
    #####type1--image type2--label
    if type==1:
        output=zoom(input, (scale, scale), order=3)
        output=(output-np.min(output))/(np.max(output)-np.min(output))
        return output
        #对output再做归一化 0-1归一化
    elif type==2:
        return zoom(input, (scale, scale), order=0)
    else:
        print('False')
def cal_centerofmass(input,method=1):
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
                if input[i, j] == 1:
                    sumx = sumx + i
                    sumy = sumy + j
                    area = area + 1
        return int(sumx / area), int(sumy / area)
def cal_cropbox(midX,midY,size,cropsize):
    assert cropsize<size
    startX=midX - int(cropsize / 2)
    startY =midY - int(cropsize / 2)
    if midX - int(cropsize / 2)<0:
        startX=0
    if midY - int(cropsize / 2)<0:
        startY=0
    if midX + int(cropsize / 2)>size:
        startX=size-cropsize
    if midY + int(cropsize / 2)>size:
        startY=size-cropsize
    return startX,startY
def curve_show(x,y,clabel=[],x_axis_label='',y_axis_label='',title='',xlim=[],ylim=[],save_dir=''):
    index_flag=True
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(dpi=200)
    assert len(x)==len(y)
    plt_title = title
    plt.title(plt_title)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if index_flag:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        x_major_locator = MultipleLocator(1)
        plt.gca().xaxis.set_major_locator(x_major_locator)
    if not len(xlim)==0:
        plt.xlim(xlim[0], xlim[1])
        #plt.ylim(ylim[0], ylim[1])
    if not len(clabel)==0:
        for i in range(len(x)):
            plt.plot(x[i], y[i], c=color_list[i], label=clabel[i])
    else:
        for i in range(len(x)):
            plt.plot(x[i], y[i], c=color_list[i])
    plt.yscale('log')
    plt.legend()
    plt.grid(linestyle=":")

    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

def draw_scatter(x,y):
    plt.xlabel('Patient')
    plt.ylabel('Foreground Dice Score')
    plt.xlim(xmax=16, xmin=0)
    plt.ylim(ymax=1, ymin=0)
    area=np.pi*4**3
    colors1 = '#00CED1'
    plt.scatter(x,y,s=area,c=colors1,alpha=0.4,label='前景(3类)dice')
    plt.show()
def draw_3class_scatter(x1,y1,x2,y2,x3,y3):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('Patient')
    plt.ylabel('Foreground Dice Score')
    plt.xlim(xmax=16, xmin=0)
    plt.ylim(ymax=1, ymin=0)
    area=np.pi*4**2
    colors1 = '#00CED1'
    colors2 = '#DC143C'
    colors3='#000000'
    plt.scatter(x1,y1,s=area,c=colors1,alpha=0.4,label='Baseline-mean-59.6%')
    plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='数据增强-mean-65.4%')
    plt.scatter(x3, y3, s=area, c=colors3, alpha=0.4, label='数据增强&定位-mean-66.9%')

    plt.show()
def combine_imglabel(image,label,draw_crop_flag=False,save_flag=False,save_dir='',crop_param=[],dye_flag=False):
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)

    if dye_flag:
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                if label[j, k] == 1:
                    #image[j, k, 0] = 255
                    image[j, k, 1] = 0
                    image[j, k, 2] = 0
                elif label[j, k] == 2:
                    image[j, k, 0] = 0
                    #image[j, k, 1] = 255
                    image[j, k, 2] = 0
                elif label[j, k] == 3:
                    image[j, k, 0] = 0
                    image[j, k, 1] = 0
                elif label[j, k] == 4:
                    image[j, k, 2] = 0
                elif label[j, k] == 5:
                    image[j, k, 1] = 0
                elif label[j, k] == 6:
                    image[j, k, 0] = 0

                    #image[j, k, 2] = 255
    else:
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                if label[j, k] == 1:
                    image[j, k, 0] = 255
                    image[j, k, 1] = 0
                    image[j, k, 2] = 0
                elif label[j, k] == 2:
                    image[j, k, 0] = 0
                    image[j, k, 1] = 255
                    image[j, k, 2] = 0
                elif label[j, k] == 3:
                    image[j, k, 0] = 0
                    image[j, k, 1] = 0
                    image[j, k, 2] = 255
                elif label[j, k] == 4:
                    image[j, k, 0] = 255
                    image[j, k, 1] = 255
                    image[j, k, 2] = 0

    if draw_crop_flag:
        startX=crop_param[0]
        startY=crop_param[1]
        size =crop_param[2]
        for j in range(startX, startX + size):
            image[j, startY, 0] = 234
            image[j, startY, 1] = 85
            image[j, startY, 2] = 32
            image[j, startY + size - 1, 0] = 234
            image[j, startY + size - 1, 1] = 85
            image[j, startY + size - 1, 2] = 32
        for k in range(startY, startY + size):
            image[startX, k, 0] = 234
            image[startX, k, 1] = 85
            image[startX, k, 2] = 32
            image[startX + size - 1, k, 0] = 234
            image[startX + size - 1, k, 1] = 85
            image[startX + size - 1, k, 2] = 32

    out=Image.fromarray(image, 'RGB')
    if save_flag:
        out.save(save_dir)
    else:
#返回Image类
        return out



def combine_imglabel_dye_previous_roi(image,label,roi):
    #对其他区域进行暗淡化4-6开
    #主要用于RPM有效性的说明。
    lower_range=0.25
    upper_range=0.4

    posROI=np.where(roi>0)
    posBG=np.where(roi==0)

    pixelROI=image[posROI]
    pixelBG = image[posBG]

    # pixelROI=normalization(pixelROI.copy(),value_max=0.7,value_min=upper_range)
    pixelBG = normalization(pixelBG.copy(),value_max=lower_range,value_min=0)

    image[posROI]=pixelROI
    image[posBG] = pixelBG


    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 1] = 0
                image[j, k, 2] = 0
            if label[j, k] == 2:
                image[j, k, 0] = 0
                image[j, k, 2] = 0
            if label[j, k] == 3:
                image[j, k, 0] = 0
                image[j, k, 1] = 0
            # if label[j, k] == 4:
            #     image[j, k, 2] = 0
    # 返回Image类
    return Image.fromarray(image, 'RGB')

def combine_imglabel_dye(image,label,saveflag=False,save_path='',des_img_size=400,crop_img_size=100,scale=2,midX=-1,midY=-1):
    #是否显示FN与FP？
    if midX>0 and midY>0:
        midX=midX
        midY=midY
    else:
        if np.sum(label) == 0:
            # 此时不做
            midX = int(label.shape[0] / 2)
            midY = int(label.shape[1] / 2)
        else:
            # 这里是求了一个质心是吧？对求质心操作进行细化

            midX, midY = cal_centerofmass(label)
    startX,startY=cal_cropbox(midX,midY,image.shape[0],crop_img_size)
    #分别做一个scale_up再合并
    upscaled_label=interpolate(label[startX:startX+crop_img_size,startY:startY+crop_img_size].copy(),2,type=2)
    upscaled_image=np.zeros_like(upscaled_label)
    upscaled_image *= 255
    upscaled_image = upscaled_image.astype(np.uint8)
    upscaled_image = upscaled_image.reshape(upscaled_image.shape[0], upscaled_image.shape[1], 1)
    upscaled_image = np.repeat(upscaled_image, 3, 2)
    for j in range(upscaled_image.shape[0]):
        for k in range(upscaled_image.shape[1]):
            if upscaled_label[j, k] == 1:
                upscaled_image[j, k, 0] = 255
                upscaled_image[j, k, 1] = 0
                upscaled_image[j, k, 2] = 0
            elif upscaled_label[j, k] == 2:
                upscaled_image[j, k, 0] = 0
                upscaled_image[j, k, 1] = 255
                upscaled_image[j, k, 2] = 0
            elif upscaled_label[j, k] == 3:
                upscaled_image[j, k, 0] = 0
                upscaled_image[j, k, 1] = 0
                upscaled_image[j, k, 2] = 255
            elif upscaled_label[j, k] == 4:
                upscaled_image[j, k, 0] = 255
                upscaled_image[j, k, 1] = 255
                upscaled_image[j, k, 2] = 0
    image *= 255
    image = image.astype(np.uint8)
    pad_image=np.pad(image,(0,des_img_size-image.shape[0]),'constant',constant_values=255)


    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)

    pad_image = pad_image.reshape(pad_image.shape[0], pad_image.shape[1], 1)
    pad_image = np.repeat(pad_image, 3, 2)

    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 1] = 0
                image[j, k, 2] = 0
            elif label[j, k] == 2:
                image[j, k, 0] = 0
                image[j, k, 2] = 0
            elif label[j, k] == 3:
                image[j, k, 0] = 0
                image[j, k, 1] = 0
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                pad_image[j, k, 1] = 0
                pad_image[j, k, 2] = 0
            elif label[j, k] == 2:
                pad_image[j, k, 0] = 0
                pad_image[j, k, 2] = 0
            elif label[j, k] == 3:
                pad_image[j, k, 0] = 0
                pad_image[j, k, 1] = 0
            elif label[j, k] == 4:
                pad_image[j, k, 2] = 0
    pad_image[pad_image.shape[0]-crop_img_size*scale:pad_image.shape[0],pad_image.shape[1]-crop_img_size*scale:pad_image.shape[1]]=upscaled_image
    output=Image.fromarray(image, 'RGB')
    output_pad=Image.fromarray(pad_image, 'RGB')
    output=output_pad
    if saveflag:
        output.save(save_path)
    else:
        return output
def combine_imglabel_dye_previous(image,label):
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 1] = 0
                image[j, k, 2] = 0
            if label[j, k] == 2:
                image[j, k, 0] = 0
                image[j, k, 2] = 0
            if label[j, k] == 3:
                image[j, k, 0] = 0
                image[j, k, 1] = 0
            if label[j, k] == 4:
                image[j, k, 2] = 0
    # 返回Image类
    return Image.fromarray(image, 'RGB')
def combine_imglabel_dye_new(image,label,saveflag=False,save_path='',draw_crop_flag=True,des_img_size=400,crop_img_size=100,scale=2):
    if np.sum(label)==0:
        #此时不做
        midX=int(label.shape[0]/2)
        midY = int(label.shape[1] / 2)
    else:
        midX, midY = cal_centerofmass(label)
    startX,startY=cal_cropbox(midX,midY,image.shape[0],crop_img_size)
    #分别做一个scale_up再合并
    upscaled_image=interpolate(image[startX:startX+crop_img_size,startY:startY+crop_img_size].copy(),2,type=1)#将image也进行上采样一倍的操作
    upscaled_label=interpolate(label[startX:startX+crop_img_size,startY:startY+crop_img_size].copy(),2,type=2)
    upscaled_image *= 255
    upscaled_image = upscaled_image.astype(np.uint8)
    upscaled_image = upscaled_image.reshape(upscaled_image.shape[0], upscaled_image.shape[1], 1)
    upscaled_image = np.repeat(upscaled_image, 3, 2)
    for j in range(upscaled_image.shape[0]):
        for k in range(upscaled_image.shape[1]):
            if upscaled_label[j, k] == 1:
                upscaled_image[j, k, 1] = 0
                upscaled_image[j, k, 2] = 0
            elif upscaled_label[j, k] == 2:
                upscaled_image[j, k, 0] = 0
                upscaled_image[j, k, 2] = 0
            elif upscaled_label[j, k] == 3:
                upscaled_image[j, k, 0] = 0
                upscaled_image[j, k, 1] = 0
    if draw_crop_flag:#200x200三道杠！
        startX=0
        startY=0
        size =200
        for j in range(startX, startX + size):
            upscaled_image[j, startY, 0] = 255
            upscaled_image[j, startY, 1] = 0
            upscaled_image[j, startY, 2] = 0
            upscaled_image[j, startY + size - 1, 0] = 255
            upscaled_image[j, startY + size - 1, 1] = 0
            upscaled_image[j, startY + size - 1, 2] = 0
        for k in range(startY, startY + size):
            upscaled_image[startX, k, 0] = 255
            upscaled_image[startX, k, 1] = 0
            upscaled_image[startX, k, 2] = 0
            upscaled_image[startX + size - 1, k, 0] = 255
            upscaled_image[startX + size - 1, k, 1] = 0
            upscaled_image[startX + size - 1, k, 2] = 0

        for j in range(startX, startX + size):
            upscaled_image[j, startY+1, 0] = 255
            upscaled_image[j, startY+1, 1] = 0
            upscaled_image[j, startY+1, 2] = 0
            upscaled_image[j, startY + size - 2, 0] = 255
            upscaled_image[j, startY + size - 2, 1] = 0
            upscaled_image[j, startY + size - 2, 2] = 0
        for k in range(startY, startY + size):
            upscaled_image[startX+1, k, 0] = 255
            upscaled_image[startX+1, k, 1] = 0
            upscaled_image[startX+1, k, 2] = 0
            upscaled_image[startX + size - 2, k, 0] = 255
            upscaled_image[startX + size - 2, k, 1] = 0
            upscaled_image[startX + size - 2, k, 2] = 0
        for j in range(startX, startX + size):
            upscaled_image[j, startY+2, 0] = 255
            upscaled_image[j, startY+2, 1] = 0
            upscaled_image[j, startY+2, 2] = 0
            upscaled_image[j, startY + size - 3, 0] = 255
            upscaled_image[j, startY + size - 3, 1] = 0
            upscaled_image[j, startY + size - 3, 2] = 0
        for k in range(startY, startY + size):
            upscaled_image[startX+2, k, 0] = 255
            upscaled_image[startX+2, k, 1] = 0
            upscaled_image[startX+2, k, 2] = 0
            upscaled_image[startX + size - 3, k, 0] = 255
            upscaled_image[startX + size - 3, k, 1] = 0
            upscaled_image[startX + size - 3, k, 2] = 0

    #在最外层加上一层红色的边框
    image *= 255
    image = image.astype(np.uint8)
    pad_image=np.pad(image,(0,des_img_size-image.shape[0]),'constant',constant_values=255)

    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)

    pad_image = pad_image.reshape(pad_image.shape[0], pad_image.shape[1], 1)
    pad_image = np.repeat(pad_image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 1] = 0
                image[j, k, 2] = 0
            elif label[j, k] == 2:
                image[j, k, 0] = 0
                image[j, k, 2] = 0
            elif label[j, k] == 3:
                image[j, k, 0] = 0
                image[j, k, 1] = 0

    pad_image[pad_image.shape[0]-crop_img_size*scale:pad_image.shape[0],pad_image.shape[1]-crop_img_size*scale:pad_image.shape[1]]=upscaled_image

    output=Image.fromarray(image, 'RGB')
    output_pad=Image.fromarray(pad_image, 'RGB')
    output=output_pad
    if saveflag:
        output.save(save_path)
    else:
        return output

def combine_img_glabel(image,label):
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 0] = 0
                image[j, k, 1] = 255
                image[j, k, 2] = 0

#返回Image类
    return Image.fromarray(image, 'RGB')
def combine_imgnewlabel(image,label):
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 0] = 255
                image[j, k, 1] = 0
                image[j, k, 2] = 0
            elif label[j, k] == 2:
                image[j, k, 0] = 0
                image[j, k, 1] = 0
                image[j, k, 2] = 255
#返回Image类
    return Image.fromarray(image, 'RGB')
def combine_cropped_imglabel(image,label,startX,startY,size):
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 0] = 255
                image[j, k, 1] = 0
                image[j, k, 2] = 0
            elif label[j, k] == 2:
                image[j, k, 0] = 0
                image[j, k, 1] = 255
                image[j, k, 2] = 0
            elif label[j, k] == 3:
                image[j, k, 0] = 0
                image[j, k, 1] = 0
                image[j, k, 2] = 255


#画出框框
    for j in range(startX,startX+size):
        image[j,startY,0]=234
        image[j,startY,1]=85
        image[j,startY,2]=32
        image[j,startY+size-1,0]=234
        image[j,startY+size-1,1]=85
        image[j,startY+size-1,2]=32
    for k in range(startY, startY + size):
        image[startX,k,0]=234
        image[startX,k,1]=85
        image[startX,k,2]=32
        image[startX+size-1,k,0]=234
        image[startX+size-1,k,1]=85
        image[startX+size-1,k,2]=32

#返回Image类
    return Image.fromarray(image, 'RGB')
def combine_cropped_imgnewlabel(image,label,startX,startY,size):
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if label[j, k] == 1:
                image[j, k, 0] = 255
                image[j, k, 1] = 0
                image[j, k, 2] = 0
            elif label[j, k] == 2:
                image[j, k, 0] = 0
                image[j, k, 1] = 0
                image[j, k, 2] = 255

#画出框框
    for j in range(startX,startX+size):
        image[j,startY,0]=234
        image[j,startY,1]=85
        image[j,startY,2]=32
        image[j,startY+size-1,0]=234
        image[j,startY+size-1,1]=85
        image[j,startY+size-1,2]=32
    for k in range(startY, startY + size):
        image[startX,k,0]=234
        image[startX,k,1]=85
        image[startX,k,2]=32
        image[startX+size-1,k,0]=234
        image[startX+size-1,k,1]=85
        image[startX+size-1,k,2]=32

#返回Image类
    return Image.fromarray(image, 'RGB')
def B_show(image,true,pred,dye_flag=False):
    #对于binary seg 结果的展示
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    if dye_flag:
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                if true[j, k] == 1 and pred[j, k] == 1:
                    #image[j, k, 0] = 255
                    image[j, k, 1] = 0
                    #image[j, k, 2] = 255
                elif true[j, k] == 1 and pred[j, k] == 0:
                    image[j, k, 0] = 0
                    #image[j, k, 1] = 255
                    #image[j, k, 2] = 255
                elif true[j, k] == 0 and pred[j, k] == 1:
                    #image[j, k, 0] = 255
                    #image[j, k, 1] = 255
                    image[j, k, 2] = 0
        return Image.fromarray(image, 'RGB')
    else:
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                if true[j, k] == 1 and pred[j, k] == 1:
                    image[j, k, 0] = 255
                    image[j, k, 1] = 0
                    image[j, k, 2] = 255
                elif true[j, k] == 1 and pred[j, k] == 0:
                    image[j, k, 0] = 0
                    image[j, k, 1] = 255
                    image[j, k, 2] = 255
                elif true[j, k] == 0 and pred[j, k] == 1:
                    image[j, k, 0] = 255
                    image[j, k, 1] = 255
                    image[j, k, 2] = 0
        return Image.fromarray(image, 'RGB')

def B_cropped_show(image,true,pred,startX,startY,size):
    #对于binary seg 结果的展示
    image *= 255
    image = image.astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = np.repeat(image, 3, 2)
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            if true[j,k]==1 and pred[j,k]==1:
                image[j, k, 0] = 255
                image[j, k, 1] = 0
                image[j, k, 2] = 255
            elif true[j,k]==1 and pred[j,k]==0:
                image[j, k, 0] = 0
                image[j, k, 1] = 255
                image[j, k, 2] = 255
            elif true[j,k]==0 and pred[j,k]==1:
                image[j, k, 0] = 255
                image[j, k, 1] = 255
                image[j, k, 2] = 0

    for j in range(startX,startX+size):
        image[j,startY,0]=234
        image[j,startY,1]=85
        image[j,startY,2]=32
        image[j,startY+size-1,0]=234
        image[j,startY+size-1,1]=85
        image[j,startY+size-1,2]=32
    for k in range(startY, startY + size):
        image[startX,k,0]=234
        image[startX,k,1]=85
        image[startX,k,2]=32
        image[startX+size-1,k,0]=234
        image[startX+size-1,k,1]=85
        image[startX+size-1,k,2]=32
    return Image.fromarray(image, 'RGB')
def image(img,saveflag=False,save_path=''):
    img*=255
    img=img.astype(np.uint8)
    output=Image.fromarray(img,'L')
    if saveflag:
        output.save(save_path)
    else:
        return output
def label_show(label,Bflag=False,saveflag=False,save_path=''):
    if Bflag:
        label=np.where(label>0,1,0)
        label *= 255
        label = label.astype(np.uint8)
        # 返回Image类
        saveimg=Image.fromarray(label, 'L')
    else:
        image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                if label[j, k] == 1:
                    image[j, k, 0] = 255
                    image[j, k, 1] = 0
                    image[j, k, 2] = 0
                elif label[j, k] == 2:
                    image[j, k, 0] = 0
                    image[j, k, 1] = 255
                    image[j, k, 2] = 0
                elif label[j, k] == 3:
                    image[j, k, 0] = 0
                    image[j, k, 1] = 0
                    image[j, k, 2] = 255
                elif label[j, k] == 4:
                    image[j, k, 0] = 255
                    image[j, k, 1] = 255
                    image[j, k, 2] = 0
        saveimg = Image.fromarray(image, 'RGB')
    if saveflag:
        #进行save
        saveimg.save(save_path)
    return saveimg
def combine_imgbinarylabel(image,label):
    X,Y=np.where(label>0)
    minX=min(X)
    maxX=max(X)
    minY=min(Y)
    maxY=max(Y)
    image *= 255
    image = image.astype(np.uint8)
    for i in range(minX):
        image[i,:]=0
    for i in range(maxX+1,image.shape[0]):
        image[i,:]=0
    for i in range(minY):
        image[:,i]=0
    for i in range(maxY+1,image.shape[1]):
        image[:,i]=0
    return Image.fromarray(image, 'L')

#返回Image类
    return Image.fromarray(label, 'L')
def draw_box_plot(data):
    plt.figure(dpi=150)
    sns.boxplot(x='method',y='dice_score',hue='type',data=data)
    plt.show()
def dataset_show(srcpath,despath):
    path = srcpath
    despath=despath
    iset = os.listdir(path[0])
    for i in iset:
        name = i[:-4]
        label = np.load(path[1]  + '/' + i)
        data = np.load(path[0]  + '/' + i)
        combined = combine_imglabel(data.copy(), label)

        original_image = image(data.copy())
        X,Y=data.shape
        height = X
        width = Y
        result = Image.new('RGB', (width * len(path), height))
        result.paste(original_image, box=(0, 0))
        result.paste(combined, box=(width, 0))
        if len(path)==3:
            roi = np.load(path[2] + '/' + i)
            combined_roi = combine_imglabel(data.copy(), roi)
            result.paste(combined_roi, box=(width*2, 0))
        result.save(despath + '/' + name + '.jpg')



def M2D3():#输出所有的图像与标签，，256x256除外2月3日，for paper
    path='G:/Task_M2D25/rawdata'
    s=os.listdir(path)
    cnt=0
    for i in s:
        pset=os.listdir(path+'/'+i)
        for p in pset:
            imgset = os.listdir(path + '/' + i+'/'+p+'/PSIR')
            for img in imgset:
                if img.endswith('.dcm'):  # 此处是图像
                    id = img[:-4]
                    ds = pydicom.read_file(path + '/' + i + '/' + p + '/PSIR/' + img)#图像进行处理
                    ds = ds.pixel_array
                    ds=(ds-np.min(ds))/(np.max(ds)-np.min(ds))
                    lb = Image.open(path + '/' + i + '/' + p + '/PSIR/' + id + '.bmp')
                    lb = np.array(lb)
                    if not lb.shape[0]==336:
                        continue
                    lb[np.where(lb == 2)] = 3  # blue
                    lb[np.where(lb == 1)] = 2  # green
                    lb[np.where(lb == 0)] = 1  # red
                    lb[np.where(lb == 16)] = 0
                    # 有可能要引入dilate ROI的概念
                    out=combine_imglabel_dye_new(ds,lb,saveflag=True,save_path='M2D3show/'+str(cnt)+'.png',)
                    cnt+=1
def M2D8():#输出所有的图像与绿色标签，，256x256除外2月9日，for paper
    path='G:/Task_M2D25/rawdata'
    s=os.listdir(path)
    cnt=0
    for i in s:
        pset=os.listdir(path+'/'+i)
        for p in pset:
            imgset = os.listdir(path + '/' + i+'/'+p+'/PSIR')
            for img in imgset:
                if img.endswith('.dcm'):  # 此处是图像
                    id = img[:-4]
                    ds = pydicom.read_file(path + '/' + i + '/' + p + '/PSIR/' + img)#图像进行处理
                    ds = ds.pixel_array
                    ds=(ds-np.min(ds))/(np.max(ds)-np.min(ds))
                    lb = Image.open(path + '/' + i + '/' + p + '/PSIR/' + id + '.bmp')
                    lb = np.array(lb)
                    if not lb.shape[0]==336:
                        continue
                    lb[np.where(lb == 2)] = 3  # blue
                    lb[np.where(lb == 1)] = 2  # green
                    lb[np.where(lb == 0)] = 1  # red
                    lb[np.where(lb == 16)] = 0
                    lb=np.where(lb==2,2,0)
                    # 有可能要引入dilate ROI的概念
                    out=combine_imglabel_dye_previous(ds,lb)
                    out.save('M2D9show/'+str(cnt)+'.png')
                    cnt+=1
if __name__=="__main__":

    print('')



    # dataset_show('../data/train','../data/train/show')
    # d1=pd.read_csv('model1.csv')
    # d2=pd.read_csv('model2.csv')
    # d3=pd.read_csv('model3.csv')
    # d4=pd.read_csv('model4.csv')
    # data=pd.concat([d1,d2,d3,d4])
    # for i in range(48*4):
    #     if data.at[str(i),'type']==0:
    #         data.at[str(i),'type'] = 'foreground DSC'
    #     if data.at[str(i),'type'] == 1:
    #         data.at[str(i),'type'] = 'mean DSC'
    #     if data.at[str(i),'type'] == 2:
    #         data.at[str(i),'type'] = 'green DSC'
    #     if data.at[str(i),'method']==1:
    #         data.at[str(i),'method'] = 'A1&B1'
    #     if data.at[str(i),'method']==2:
    #         data.at[str(i),'method']= 'A1&B2'
    #     if data.at[str(i),'method']==3:
    #         data.at[str(i),'method'] = 'A1&B3'
    #     if data.at[str(i),'method']==4:
    #         data.at[str(i),'method'] = 'A2&B1'
    #
    # draw_box_plot(data)
    # y1=np.load('../seg2D/M1/seg.npy')
    # y2=np.load('../seg2D/M1/ED_seg.npy')
    # y3=np.load('../seg2D/M1/ED_crop_seg.npy')
    # x1=np.array(range(1,17))
    # x2 = np.array(range(1, 17))
    # x3 = np.array(range(1, 17))
    # draw_3class_scatter(x1,y1,x2,y2,x3,y3)
    #
    # print('')
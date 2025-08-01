import argparse
import os
import torch
from PIL import ImageDraw
import numpy as np
from Utils.Dataset import DirectSeg_Basic_cropped_Dataset_noaug
from Utils.show import combine_imglabel,B_show
from Utils.show import image as IM
from Utils.dice import dice_coeff,B_recall_g_coeff
from PIL import Image
import cv2
from Utils.Dataset import BasicDataset_3D
from skimage import measure
from scipy.ndimage.interpolation import zoom
import nibabel as nib

def process(input,method=2,iterations=5):
    #输入应该为0,255二值图像 numpy数组
    input=input.astype(np.uint8)
    assert len(input.shape)==2
    if method==2:
        denoised_input = input
    else:
        print('F')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
    output = cv2.dilate(denoised_input, kernel, iterations=iterations)
    return output

def interpolate(input,scale,method):
    return zoom(input, (scale,scale), order=method)

def niipred_2np(nii_path,np_path):
    pset=os.listdir(nii_path)
    #直接全部保存为3D
    try:
        os.makedirs(np_path)
    except OSError:
        pass

    for p in pset:
        if p.endswith('.nii.gz'):#进行处理
            name=p[5:8]
            name=int(name)
            pred=nib.load(nii_path+'/'+p)
            pred=pred.get_fdata()#3D!
            #是否需要维度转换
            pred=np.swapaxes(pred,0,1)#交换第一个维度
            # if pred.shape[0]==256:#此时进行插值
            #     interpolate_pred=np.zeros((336,336,pred.shape[2]))
            #     for i in range(pred.shape[2]):
            #         slice_pred=pred[:,:,i].copy()
            #         slice_pred=slice_pred.astype(int)
            #         slice_pred=interpolate(slice_pred,336/256,method=0)#将分割结果进行最邻近插值操作
            #         interpolate_pred[:,:,i]=slice_pred
            #     pred=interpolate_pred
            #是否需要对尺寸不足的进行插值
            pred=pred.astype(np.uint8)
            np.save(np_path+'/%d.npy'%(name),pred)#可能有256x256的预测结果
    #将nii的预测结果全部放入numpy容器中
def normalization(input):
    return (input-np.min(input))/(np.max(input)-np.min(input))
def cal_np_recall(roi,truth,epsilon=1e-6):
    inter = np.dot(roi.reshape(-1), truth.reshape(-1))  # 取交集
    sets_sum = np.sum(truth)
    return (inter + epsilon) / (sets_sum + epsilon)
def post_process_2D(input,iterations=5):
    #判断进行2D后处理还是3D后处理
    x,y=input.shape
    #对于 筛除假阳性的点 大多数是对于tensor输入值
    tmp=np.where(input>0,1,0)
    #进行膨胀操作
    tmp=tmp.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
    tmp = cv2.dilate(tmp, kernel, iterations=iterations)

    labels,num=measure.label(tmp,return_num=True)
    max_num=-1
    for i in range(num):
        posX,posY=np.where(labels==(i+1))
        if len(posX)>max_num:
            max_num=len(posX)
    for i in range(num):
        posX, posY = np.where(labels == (i + 1))
        if len(posX) < max_num:
            # 此时被判定为假阳性区域，筛除！
            tmp[posX, posY] = 0
    tmp=tmp.astype(int)
    return input*tmp.reshape(x,y)


def roi_post_process_new(input,threshold=300):
    #判断进行2D后处理还是3D后处理,改为只保留最大联通区域！！
    x,y=input.shape
    tmp=np.where(input>0,1,0)
    dilated_tmp=process((tmp.astype(np.uint8)).copy(), method=2, iterations=3)#
    dilated_tmp=dilated_tmp.astype(int)
    labels,num=measure.label(dilated_tmp,return_num=True)
    max_num=-1
    for i in range(num):
        posX,posY=np.where(labels==(i+1))
        if len(posX)>max_num:
            #此时被判定为假阳性区域，筛除！
            max_num=len(posX)
    for i in range(num):
        posX, posY = np.where(labels == (i + 1))
        if len(posX) < max_num:
            # 此时被判定为假阳性区域，筛除！
            dilated_tmp[posX, posY] = 0

    input=input * dilated_tmp.reshape(x, y)

    x,y=input.shape
    tmp=np.where(input>0,1,0)
    #进行膨胀操作
    labels,num=measure.label(tmp,return_num=True)
    for i in range(num):
        posX,posY=np.where(labels==(i+1))
        if len(posX)<threshold:
            #此时被判定为假阳性区域，筛除！
            tmp[posX,posY]=0
    tmp=tmp.astype(int)
    input=input * tmp.reshape(x, y)
    return   input

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


#尝试进一步数据增广


def ROI_generate_2D_ideal(
              dataloader,
              des_path=[],
              pre_size=256,
              crop_size=128,
              show=True,
              show_path='',
              itr=5,
              RD_flag=False,
              save_flag=False,
              ideal_flag=False,
              norm_flag=False,
              length=0


):

    assert ideal_flag==True
    ##need to save masked_img and masked_label and masked_roi
################################
    mean_g_recall = 0
    mean_B_recall=0
    mean_B_dice=0
    recall_num = 0
    # 计算类别之间的数量比值！
    if itr==0:
        threshold=50
    if itr==2:
        threshold=100
    if itr==5:
        threshold=200
    if itr==12:
        threshold=450
    if itr==16:
        threshold=620
    if itr==8:
        threshold=300
    if itr == 10:
        threshold = 375
    if itr == 14:
        threshold = 550
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print('----------------------------')
            img, label, index = data #presize(256)Xpresize(256)
            point_list = []

            #pred=pred*(Broi.cuda())#pred是否需要后处理
            for j in range(img.shape[0]):
                name = dataloader.dataset.atlas_list[index[j].item()]
                print(name)
                name_idx=name[:-4]
                name_idx=int(name_idx)

                slice_image = img[j, 0]
                slice_label = label[j]  # wxh
                slice_image = slice_image.numpy()
                slice_label = slice_label.numpy()
                if ideal_flag:#此时ROI是完全理想的
                    ring_label=np.where(slice_label==4,0,slice_label)
                    B_pred=np.where(ring_label>0,1,0)
                B_pred = B_pred.astype(np.uint8)#256x256

                  # 2D slice 此处对256x256进行randomaugment
                roi = process(B_pred.copy(), method=2, iterations=itr)  # roi---binary2D

                #对ROI进行后处理操作
                roi=roi_post_process_new(roi,threshold=threshold)#256x256

                slice_g_label = np.where(slice_label == 2, 1, 0)
                #逐slice计算RG DSC
                slice_b_label = np.where(slice_label ==4, 0, slice_label)
                slice_b_label = np.where(slice_b_label >0, 1, 0)


                score_dilated_G_recall = (B_recall_g_coeff((torch.from_numpy(roi)).float(),
                                            (torch.from_numpy(slice_g_label)).float())).item()
                score_dilated_B_recall = (B_recall_g_coeff((torch.from_numpy(roi)).float(),
                                            (torch.from_numpy(slice_b_label)).float())).item()
                mean_B_recall+=score_dilated_B_recall

                if np.sum(roi) == 0 and np.sum(slice_g_label)==0:  # 说明此slice中并不含有R&G区域,
                    #print('something wrong!!!!')
                    continue
                if not np.sum(slice_g_label) == 0:
                    score_dilated_G_recall = (B_recall_g_coeff((torch.from_numpy(roi)).float(),
                                                               (torch.from_numpy(slice_g_label)).float())).item()
                    mean_g_recall += score_dilated_G_recall
                    recall_num += 1
                #将256x256进行裁剪
                if not np.sum(roi) == 0:
                    midX, midY = cal_centerofmass(roi.copy())
                else:
                    midX=int((slice_image.shape[0])/2)
                    midY=int((slice_image.shape[0])/2)
                pos = [midX, midY]
                point_list.append(pos)
                startX, startY = cal_cropbox(midX, midY, pre_size, crop_size)#256,128

                roi = roi[startX:startX + crop_size, startY:startY + crop_size]#128x128
                slice_label = slice_label[startX:startX + crop_size, startY:startY + crop_size]
                slice_image = slice_image[startX:startX + crop_size, startY:startY + crop_size]

                if norm_flag:
                    slice_image=normalization(slice_image.copy())

                masked_image =  slice_image#以后都这么进行操作
                masked_label =  slice_label

                if RD_flag:
                    name=str(name_idx+length)+'.npy'
                if save_flag:
                    np.save(des_path[0] + '/' + name, masked_image)
                    np.save(des_path[1] + '/' + name, masked_label)
                    np.save(des_path[2] + '/' + name, roi)
                if show:

                    raw_image = IM(slice_image.copy())
                    raw_label = combine_imglabel(slice_image.copy(), slice_label.copy(),dye_flag=True)
                    combined_label = combine_imglabel((masked_image*roi).copy(), slice_label.copy())
                    height = crop_size
                    width = crop_size
                    result = Image.new('RGB', (width * 3, height))
                    result.paste(raw_image, box=(0, 0))  # 1，原图
                    result.paste(raw_label, box=(width, 0))  # 1，原图
                    result.paste(combined_label, box=(width * 2, 0))
                    #写入文字内容
                    draw = ImageDraw.Draw(result)
                    draw.text((width, 0),'d-g-rec'+str(round(score_dilated_G_recall,2)))  # g-dilated-recall
                    draw.text((width*2, 0),'d-lb-rec'+str(round(score_dilated_B_recall,2)))  # b-dilated-recall
                    result.save(show_path + '/' + name[:-4] + '.jpg')

                #need to show the S2 stage input image and label version!!!
                #R&G ROI has been created!
    print(mean_g_recall / recall_num)
    print(mean_B_dice/len(dataloader.dataset.atlas_list))
    print(mean_B_recall/len(dataloader.dataset.atlas_list))


def ROI_generate_3D(
              dataloader,
              des_path=['','',''],
              show=True,
              roi_save_path='',
              out_path='',
              itr=5,
              ideal_flag=False,
              raw_size_flag=True

):
    ##need to save masked_img and masked_label and masked_roi

    mean_lb_recall = 0
    mean_g_recall = 0
    mean_B_recall=0
    mean_B_dice=0
    recall_num = 0
    if itr==0:
        threshold=50
    if itr==2:
        threshold=100
    if itr==5:
        threshold=200
    if itr==12:
        threshold=450
    if itr==16:
        threshold=620
    if itr==8:
        threshold=300
    if itr == 10:
        threshold = 375
    if itr == 14:
        threshold = 550
    cnt=0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img, label,pred = data #presize(256)Xpresize(256)
            name=(dataloader.dataset.atlas_list[i])[:-4]
            print(name)
            assert img.shape[0]==1###逐3D进行test！！！
            assert img.shape[2] == img.shape[3]  ###逐张进行test！！！
            BS,cn,W,H,Z=img.shape

            S1_pred=pred[0]
            S1_dilate = torch.zeros(W,H,Z)


            #生成validation的
            ##################S1 Segmentation begin#########################
            ########################S1---->S2#####################################
            for j in range(Z):
                # slice_pred = ((pred[j].clone()).cpu()).numpy()  # wxh
                # S1_pred[startX:startX+s_size[0],startY:startY+s_size[0],j]=torch.from_numpy(slice_pred)
                # 对2class coarse seg进行膨胀处理，得到目的ROI
                slice_label = (label[0, :, :, j]).numpy()
                slice_image=(img[0,0,:,:,j]).numpy()
                if ideal_flag:#此时套用理想的ROI
                    slice_label = (label[0, :, :, j]).numpy()
                    slice_RGlabel=np.where(slice_label==2,1,slice_label)
                    slice_RGlabel = np.where(slice_RGlabel == 1, 1, 0)
                    S1_pred[:,:,j]=torch.from_numpy(slice_RGlabel)
                #同样对ROI进行后处理操作

                slice_pred=S1_pred[:,:,j].numpy()
                slice_pred=np.where(slice_pred==1,1,0)
                roi = process((slice_pred.astype(np.uint8)).copy(), method=2, iterations=itr)  # roi---binary2D
                roi=roi_post_process_new(roi,threshold=threshold)
                S1_dilate[:,:,j]=torch.from_numpy(roi)
                if np.sum(roi) == 0:  # 说明此slice中并不含有R&G区域
                    continue


                print('----------2D-----------')
                pre_size = W
                # 进行ROI后处理

                slice_g_label = np.where(slice_label == 2, 1, 0)
                # 逐slice计算RG DSC
                slice_b_label = np.where(slice_label == 4, 0, slice_label)
                slice_b_label = np.where(slice_b_label > 0, 1, 0)
                mean_B_dice += (dice_coeff((torch.from_numpy(slice_pred)).float(),
                                           (torch.from_numpy(slice_b_label)).float())).item()
                score_dilated_B_recall = (B_recall_g_coeff((torch.from_numpy(roi)).float(),
                                                           (torch.from_numpy(slice_b_label)).float())).item()
                mean_B_recall += score_dilated_B_recall

                if np.sum(roi) == 0 and np.sum(slice_g_label) == 0:  # 说明此slice中并不含有R&G区域,
                    # print('something wrong!!!!')
                    continue
                if not np.sum(slice_g_label) == 0:
                    score_dilated_G_recall = (B_recall_g_coeff((torch.from_numpy(roi)).float(),
                                                               (torch.from_numpy(slice_g_label)).float())).item()
                    mean_g_recall += score_dilated_G_recall
                    recall_num += 1

                if not np.sum(roi) == 0:
                    midX, midY = cal_centerofmass(roi.copy())
                else:
                    midX = int((slice_image.shape[0]) / 2)
                    midY = int((slice_image.shape[0]) / 2)
                startX, startY = cal_cropbox(midX, midY, pre_size, crop_size)  # 256,128

                if not raw_size_flag:
                    # 如果这样的话，保存336x336的ROIflag
                    roi = roi[startX:startX + crop_size, startY:startY + crop_size]  # 128x128 numpy文件
                    slice_label = slice_label[startX:startX + crop_size, startY:startY + crop_size]
                    slice_image = slice_image[startX:startX + crop_size, startY:startY + crop_size]
                    # 对ROI进行后处理操作
                    slice_pred = slice_pred[startX:startX + crop_size, startY:startY + crop_size]

                if norm_flag:
                    slice_image = normalization(slice_image.copy())

                masked_image = slice_image  # 以后都这么进行操作
                masked_label = slice_label

                if save_flag:
                    np.save(des_path[0] + '/' + str(cnt)+'.npy', masked_image)
                    np.save(des_path[1] + '/' + str(cnt)+'.npy', masked_label)
                    np.save(des_path[2] + '/' + str(cnt)+'.npy', roi)
                cnt += 1

            if not roi_save_path=='':
                #对3D ROI进行存储
                try:
                    os.makedirs(roi_save_path)
                except OSError:
                    pass
                np.save(roi_save_path+'/'+name+'.npy',S1_dilate)
            ########################metric evaluation###########################
            ########################visualization###########################分别输出三个阶段的处理结果！
            if show:
                for z in range(Z):
                    full_image = img[0, 0, :, :, z].numpy()
                    gt=label[0,:,:,z].numpy()
                    p1 = S1_pred[:, :, z].numpy()
                    pd1 = S1_dilate[:, :, z].numpy()
                    ######单张图save区

                    #原图->标签图->S1pred->S1pred dialte
                    original_image = IM(full_image.copy())
                    gt_image = combine_imglabel(full_image.copy(), gt)
                    pred_s1_image = combine_imglabel(full_image.copy(),p1)
                    pred_s1_dilate_image = combine_imglabel(full_image.copy(),pd1)
                    result = Image.new('RGB', (W * 4, H))
                    result.paste(original_image, box=(0, 0))
                    result.paste(gt_image, box=(W, 0))
                    result.paste(pred_s1_image, box=(W * 2, 0))
                    result.paste(pred_s1_dilate_image, box=(W * 3, 0))
                    result.save(out_path+'/'+name+'-'+str(z)+ '.png')
    total_num=len(dataloader)
    print(mean_g_recall/total_num)
    # print(mean_lb_recall / total_num)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--trainbatchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--valbatchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
    return parser.parse_args()

if __name__ == '__main__':

#仅需要给validation生成对应的二阶段数据集
#train不需要进行预测了
    fold_id=1
    iteration = 5
    assert iteration==5

    train2Dimagedir='../Data/np/train/auged_image'
    train2Dlabeldir = '../Data/np/train/auged_label'
    task_title = '../Data/S2'

    opt = get_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    ##########
    crop_size = 256
    train_dataset = DirectSeg_Basic_cropped_Dataset_noaug(train2Dimagedir, train2Dlabeldir, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=opt.trainbatchsize,
                                                   num_workers=opt.workers)

    # 直接进行裁剪,基于真实标签生成
    n_classes = 3
    n_channels = 1
    des_img_path = task_title + '/train_ideal/image'
    des_label_path = task_title + '/train_ideal/label'
    des_roi_path = task_title + '/train_ideal/roi'
    show_path = task_title + '/train_ideal/show'
    try:
        os.makedirs(des_img_path)
    except OSError:
        pass
    try:
        os.makedirs(des_label_path)
    except OSError:
        pass
    try:
        os.makedirs(des_roi_path)
    except OSError:
        pass
    try:
        os.makedirs(show_path)
    except OSError:
        pass

    save_flag = True
    show_flag = True
    norm_flag = True
    ideal_flag = True
    RD_flag = False
    # 在这里进行iteration的调整！

    ROI_generate_2D_ideal(train_dataloader, [des_img_path, des_label_path, des_roi_path], crop_size,
                    128,
                    show=show_flag, show_path=show_path, itr=iteration, RD_flag=RD_flag,
                    save_flag=save_flag,
                    ideal_flag=ideal_flag, norm_flag=norm_flag
                    )

    #######validation 2D 3D,验证集可以生成原始图像尺寸大小的，没有问题#或者不生成原尺寸的也无所谓

#不对进行裁剪

#####################3D的生成############################################################


    des_img_path = task_title + '/val/image'
    des_label_path = task_title + '/val/label'
    des_roi_path = task_title + '/val/roi'
    show_path = task_title + '/val/show'
    des_3Droi_path = task_title + '/val/roi3D'

    try:
        os.makedirs(des_img_path)
    except OSError:
        pass
    try:
        os.makedirs(des_label_path)
    except OSError:
        pass
    try:
        os.makedirs(des_roi_path)
    except OSError:
        pass
    try:
        os.makedirs(show_path)
    except OSError:
        pass
    try:
        os.makedirs(des_3Droi_path)
    except OSError:
        pass

    test3D_niipreddir = '../Data/nii/nnUNet_S1/nnUNet_raw_data_base/nnUNet_raw_data/Task001_MI/inferVal'
    test3Dimagedir = '../Data/np/val/image3D'
    test3Dlabeldir = '../Data/np/val/label3D'
    test3Dpreddir = '../Data/np/val/S1_pred3D'
    niipred_2np(test3D_niipreddir, test3Dpreddir)
    test_dataset = BasicDataset_3D(test3Dimagedir, test3Dlabeldir,test3Dpreddir)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,
                                                  num_workers=opt.workers)

    ROI_generate_3D(test_dataloader,des_path=[des_img_path, des_label_path, des_roi_path], show=True, out_path=show_path, roi_save_path=des_3Droi_path,
                    itr=iteration, ideal_flag=False)















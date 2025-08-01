
import argparse
import os
from Model.STSN import kiSTSN_plus
import torch
from Utils.Dataset import BasicDataset_3D
import json
import numpy as np
import time
import cv2
from skimage import measure
from Utils.show import combine_imglabel_dye,image,combine_imglabel,combine_imglabel_dye_previous
import nibabel as nib

STAGE_SHOW_SIZE=160

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
def cal_centerofmass(input,method=1):
    height,width=input.shape
    sumx=0
    sumy=0
    area=0
    #在这里再进一步细化
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
def process(input,method=1,iterations=5):
    #对数据格式进行检验，看看是否适用于做dilate
    #输入应该为0,255二值图像 numpy数组
    input=input.astype(np.uint8)
    assert len(input.shape)==2
    if method==2:
        denoised_input = input
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
    output = cv2.dilate(denoised_input, kernel, iterations=iterations)
    return output
def cal_cropbox(midX,midY,size,cropsize):
    if isinstance(size,int):
        size=[size,size]
    assert cropsize<=size[0]
    assert cropsize <= size[1]
    startX=midX - int(cropsize / 2)
    startY =midY - int(cropsize / 2)
    if midX - int(cropsize / 2)<0:
        startX=0
    if midY - int(cropsize / 2)<0:
        startY=0
    if midX + int(cropsize / 2)>size[0]:
        startX=size[0]-cropsize
    if midY + int(cropsize / 2)>size[1]:
        startY=size[1]-cropsize
    return startX,startY
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
def normalization(input):
    return (input-np.min(input))/(np.max(input)-np.min(input))

def normal_test(
        net,
        dataloader,
        s_size=[],
        itr=5,
        cav_first_flag=False,
        norm_flag=True,
        testpreddir='',
        show='',

#用于一般的单模型网络
):

    net.eval()
    print('val')
    if itr==12:
        threshold=450
    if itr==16:
        threshold=620
    if itr==8:
        threshold=300
    if itr==5:
        threshold=200

    with torch.no_grad():
        start = time.time()
        for i, data in enumerate(dataloader):
            #获取index,即其对应的姓名

            img, label,S1_pred= data #presize(336)Xpresize(336)
            S1_pred=S1_pred[0]
            show_img=img.clone()
            name=(dataloader.dataset.atlas_list[i])[:-4]
            print(name)
            assert img.shape[0]==1###逐3D进行test！！！
            assert img.shape[2] == img.shape[3]  ###逐张进行test！！！
            BS,cn,W,H,Z=img.shape


            S2_pred = torch.zeros(W,H,Z)
            S1_dilate = torch.zeros(W,H,Z)#ROI


            S2_roi=torch.zeros(Z,s_size[1],s_size[1])
            S2_point_list=np.zeros((Z,2),dtype=int)
            S2_input=torch.zeros(Z,cn,s_size[1],s_size[1])

            ##################S1 Segmentation begin#########################
            ########################S1---->S2#####################################
            ##########是否需要也对心腔区域 使用最大连通域筛除算法？
            for j in range(Z):

                slice_image = (img[0,0,:,:,j]).numpy()
                slice_pred=S1_pred[:,:,j].numpy()
                slice_pred=np.where(slice_pred==1,1,0)
                roi = process((slice_pred.astype(np.uint8)).copy(), method=2, iterations=itr)  # roi---binary2D
                roi=roi_post_process_new(roi,threshold=threshold)

                S1_dilate[:,:,j]=torch.from_numpy(roi)
                #此处对ROI进行一系列后处理操作
                if np.sum(roi)==0:
                    continue
                S1_midX, S1_midY = cal_centerofmass(roi.copy())
                S1_startX, S1_startY = cal_cropbox(S1_midX, S1_midY, W, s_size[1])
                S2_point_list[j,0]=int(S1_startX)
                S2_point_list[j, 1] = int(S1_startY)

                if np.sum(roi) == 0:  # 说明此slice中并不含有R&G区域
                    S2_point_list[j, 0] = -1
                    S2_point_list[j, 1] = -1
                    continue
                    #此处把原图也进行一次对应的更改
                S2_roi[j] = torch.from_numpy(roi[S1_startX:S1_startX + s_size[1], S1_startY:S1_startY + s_size[1]] )
                slice_image=slice_image[S1_startX:S1_startX + s_size[1], S1_startY:S1_startY + s_size[1]]
                if norm_flag:
                    slice_image=normalization(slice_image.copy())
                masked_image = roi[S1_startX:S1_startX + s_size[1], S1_startY:S1_startY + s_size[1]] * slice_image
                if np.sum(roi) == 0:  # 说明此slice中并不含有R&G区域
                    S2_point_list[j, 0] = -1
                    S2_point_list[j, 1] = -1
                    continue
                    #此处把原图也进行一次对应的更改
                S2_roi[j] = torch.from_numpy(roi[S1_startX:S1_startX + s_size[1], S1_startY:S1_startY + s_size[1]] )
                S2_input[j, 0] = torch.from_numpy(masked_image)

            ##################S2 Segmentation begin#########################
            pred=net(S2_input.cuda())#输入128*128!!!
            pred=torch.argmax(pred,dim=1)
            pred = pred * (S2_roi.cuda())
            for j in range(Z):
                if S2_point_list[j,0]==-1:
                    continue
                S1_startX=S2_point_list[j,0]
                S1_startY = S2_point_list[j, 1]
                S2_pred[S1_startX:S1_startX+s_size[1],S1_startY:S1_startY+s_size[1],j]=((pred[j].clone()).cpu())#1，2，3
            #pred应该包含1，2，3三种预测结果
            S1_pred_cav = torch.where(S1_pred == 2, 4, 0)
            S2_pred = torch.tensor(S2_pred, dtype=torch.int64)
            # 是否需要对cav进行额外处理呢？
            if cav_first_flag:  # 此时以cavity为优先
                #应该以cavity优先
                S2_pred = torch.where(S1_pred_cav == 4, 4, S2_pred)
            else:
                S2_pred = torch.where(S2_pred > 0, S2_pred, S1_pred_cav)

                #其他心腔区域对应低优先级,如何处理额外的误诊的心腔标签？？？
            ########################metric evaluation###########################
            print('---------------------')
            masks_true=label[0]
            predict=S2_pred.clone()

            predict=predict.cpu()
            predict=torch.squeeze(predict)
            predict=predict.numpy()
            predict=predict.astype(np.uint8)

            np.save(testpreddir+'/%s.npy'%name,predict)


            if not show=='':#附加对各个阶段的展示
                #输出各个阶段的预测结果
                centroid=True
                for z in range(Z):
                    full_image = img[0, 0, :, :, z].numpy()
                    full_image=normalization(full_image)
                    gt=label[0,:,:,z].numpy()
                    p1 = S1_pred[:, :, z].numpy()
                    p2 = S2_pred[:, :, z].numpy()
                    pd1 = S1_dilate[:, :, z].numpy()
                    #原图->标签图->S1pred->S1pred dialte->S2segment
                    #适当裁剪，中心定位？

                    if centroid:
                        slice_roi = np.where(gt > 0, 1, 0)
                        show_midX, show_midY = cal_centerofmass(slice_roi.copy())
                        show_startX, show_startY = cal_cropbox(show_midX, show_midY,
                                                           [full_image.shape[0], full_image.shape[1]], STAGE_SHOW_SIZE)

                        full_image=full_image[show_startX:show_startX + STAGE_SHOW_SIZE, show_startY:show_startY + STAGE_SHOW_SIZE]
                        gt=gt[show_startX:show_startX + STAGE_SHOW_SIZE, show_startY:show_startY + STAGE_SHOW_SIZE]
                        p1=p1[show_startX:show_startX + STAGE_SHOW_SIZE, show_startY:show_startY + STAGE_SHOW_SIZE]
                        p2=p2[show_startX:show_startX + STAGE_SHOW_SIZE, show_startY:show_startY + STAGE_SHOW_SIZE]
                        pd1=pd1[show_startX:show_startX + STAGE_SHOW_SIZE, show_startY:show_startY + STAGE_SHOW_SIZE]
                        full_image = normalization(full_image)



                    original_image = image(full_image.copy())

                    gt_image = combine_imglabel(full_image.copy(), gt,dye_flag=True)#四色label
                    #心肌区域与cavity区域
                    #此处改变一下颜色
                    p1=np.where(p1==2,4,p1)
                    p1=np.where(p1==1,5,p1)

                    pred_s1_image = combine_imglabel(full_image.copy(),p1,dye_flag=True)

                    p2=np.where(p2==4,0,p2)
                    pred_s2_image = combine_imglabel(full_image.copy(), p2,dye_flag=True)


                    pd1=np.where(pd1==1,6,0)

                    pred_s1_dilate_image = combine_imglabel(full_image.copy(),pd1,dye_flag=True)
                    #染色成什么色


                    #分别进行输出保存即可

                    width=STAGE_SHOW_SIZE
                    height=STAGE_SHOW_SIZE
                    # result = Image.new('RGB', (width * 4, height))
                    # result.paste(gt_image, box=(0, 0))  # 1，原图
                    # result.paste(pred_s1_image, box=(width, 0))  # 1，原图
                    # result.paste(pred_s1_dilate_image, box=(width*2, 0))  # 2，原图带标签，0，1，2三类标签
                    # result.paste(pred_s2_image, box=(width * 3, 0))
                    # result.save(show+'/'+name+'-'+str(z) + '.png')

                    #同时分别保存

                    gt_image.save(show+'/'+name+'-'+str(z) + 'gt.png')
                    pred_s1_image.save(show + '/' + name + '-' + str(z) + 's1.png')
                    pred_s1_dilate_image.save(show + '/' + name + '-' + str(z) + 's1roi.png')
                    pred_s2_image.save(show + '/' + name + '-' + str(z) + 's2.png')


                    #输出分割过程。

            #进行展示


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--valbatchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
    return parser.parse_args()
if __name__ == '__main__':

    opt = get_args()
    print(opt)

    fold = 1

    S1nii_testpreddir = '../Data/nii/nnUNet_S1/nnUNet_raw_data_base/nnUNet_raw_data/Task00%d_MI/inferTs' % (fold)
    S1np_testpreddir = '../Data/np/test/S1_pred3D'
    niipred_2np(S1nii_testpreddir, S1np_testpreddir)
    testimagedir = '../Data/np/test/image3D'
    testlabeldir = '../Data/np/test/label3D'
    testpreddir = '../Data/np/test/Ours_pred3D'
    show_path = '../Data/np/test/show'


    try:
        os.makedirs(testpreddir)
    except OSError:
        pass
    try:
        os.makedirs(show_path)
    except OSError:
        pass


    netWeights = '../Train/record/Ours_CL'
    # 目前来说无论如何调节超参数，都是没有影响的，是不是我们的ROIshift对结果存在影响呢？？？但是如果离线数据量太大了该怎么处理？
    fset = os.listdir(netWeights)
    file_name = ''
    for f in fset:
        if f.endswith('.pth'):
            file_name = f
    netWeights = netWeights + '/' + file_name

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # 以三维为单位

    n_classes = 4
    net_trained_flag = 0
    # 默认生成器鉴别器是成对的

    net = kiSTSN_plus(class_num=n_classes, softmax_flag=True, fusion_flag=1, direction='L2H',
                fusion_mode=1)


    if netWeights != '':
        net.load_state_dict(torch.load(netWeights))
        net_trained_flag = 1
    if opt.cuda:
        net.cuda()
    net.eval()
    test_dataset = BasicDataset_3D(testimagedir, testlabeldir,S1np_testpreddir)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=opt.valbatchsize,
                                                  num_workers=opt.workers)
    norm_flag = True
    # 由于需要测量豪斯多夫距离，所以补充
    normal_test(net, test_dataloader, [256, 128], itr=5, norm_flag=norm_flag, testpreddir=testpreddir,show=show_path)





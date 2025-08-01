#用于nnLVP的结果预测
import argparse
import os
import torch
import json
import numpy as np
import nibabel as nib
import cv2
from skimage import measure
from Utils.Dataset import BasicDataset_3D
from Utils.show import combine_imglabel_dye,image,combine_imglabel,combine_imglabel_dye_previous


STAGE_SHOW_SIZE=160
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        content_all= json.load(f)
    return content_all



def mk_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def niipred_2np(nii_path,np_path):
    try:
        os.makedirs(np_path)
    except OSError:
        pass
    npset=os.listdir(np_path)
    # if not len(npset)==0:
    #     return
    pset=os.listdir(nii_path)
    for p in pset:
        if p.endswith('.nii.gz'):#进行处理
            name=p[5:9]
            name=int(name)
            pred=nib.load(nii_path+'/'+p)
            pred=pred.get_fdata()#3D!
            #有如何的对照关系？
            pred=pred[:,:,0]
            #是否需要维度转换
            pred=np.swapaxes(pred,0,1)#交换第一个维度 是否交换维度？是需要交换的 336x336
            #在这里有必要全部转为336x336
            ###此处全部都是128x128的预测
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
        dataloader,
        s_size=[],
        itr=5,
        cav_first_flag=True,
        fold_id=1,
        np_testpreddir='',
        S1_np_testpreddir='',
        testpreddir='',
        show=''


#用于一般的单模型网络
):
    #首先第一步，3D体与2D切片的对应关系
    #第二步 2D切片如何从128映射到336的切片当中去
    #其中对于Cav这一类还是要用到我们自己的方法


    content=load_json('patient_map_sliceindex_info_%d_test.json' % fold_id)

    threshold_map={0:50,2:100,5:200,12:450,16:620,8:300,10:375,14:550}
    threshold=threshold_map[itr]

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            #获取index,即其对应的姓名

            img, label = data #presize(336)Xpresize(336)


            name=(dataloader.dataset.atlas_list[i])[:-4]
            # print(name)
            assert img.shape[0]==1###逐3D进行test！！！
            assert img.shape[2] == img.shape[3]  ###逐张进行test！！！
            BS,cn,W,H,Z=img.shape
            S1_pred = torch.zeros(W,H,Z)
            S2_pred = torch.zeros(W,H,Z)
            S1_dilate = torch.zeros(W,H,Z)#ROI
            slice_index_list = content[name]

            # pred应该包含1，2，3三种预测结果

            for j in range(Z):
                slice_name = slice_index_list[j]
                slice_pred_S1 = np.load(S1_np_testpreddir+ '/%d.npy' % (slice_name))  # 336x336
                #同样也需要二维的预测结果

                slice_pred_S1 = torch.from_numpy(slice_pred_S1)
                # 先不进行相乘操作
                S1_pred[:, :, j] = slice_pred_S1  # 1，2，3


            S2_roi=torch.zeros(Z,s_size[1],s_size[1])
            S2_point_list=np.zeros((Z,2),dtype=int)

            ##################S1 Segmentation begin#########################

            ########################S1---->S2#####################################
            ##########是否需要也对心腔区域 使用最大连通域筛除算法？
            for j in range(Z):
                # 对2class coase seg进行膨胀处理，得到目的ROI
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

            ##################S2 Segmentation begin#########################

            #pred应该包含1，2，3三种预测结果

            for j in range(Z):
                slice_roi=S2_roi[j]


                slice_name=slice_index_list[j]
                if os.path.exists(np_testpreddir+'/%d.npy'%(slice_name)):
                    slice_pred = np.load(np_testpreddir + '/%d.npy' % (slice_name))  # 336x336
                else:
                    slice_pred=np.zeros((W,H))
                slice_pred=torch.from_numpy(slice_pred)
                # slice_pred=slice_pred*slice_roi

                #还要和roi相乘！说实话也没有必要，也有必要哈哈哈
                #为什么没有必要
                #先不进行相乘操作
                S2_pred[:,:,j]=slice_pred#1，2，3

            S1_pred_cav = torch.where(S1_pred == 2, 4, 0)
            S2_pred = torch.tensor(S2_pred, dtype=torch.int64)
            # 是否需要对cav进行额外处理呢？
            if cav_first_flag:  # 此时以cavity为优先
                S2_pred = torch.where(S1_pred_cav == 4, 4, S2_pred)
            else:
                S2_pred = torch.where(S2_pred > 0, S2_pred, S1_pred_cav)

                #其他心腔区域对应低优先级,如何处理额外的误诊的心腔标签？？？
            ########################metric evaluation###########################
            # print('---------------------')

            masks_true=label[0]
            predict=S2_pred.clone()

            predict_save=torch.squeeze(predict).cpu().numpy()
            if save_flag:
                np.save(testpreddir + '/%s.npy' % name, predict_save.astype(np.uint8))

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


    #可以输出一下指标


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--valbatchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
    return parser.parse_args()

if __name__ == '__main__':
    #将模型替换为我们自己的模型之后
    #需要将nnS1pred加进来
    opt = get_args()
    print(opt)
    save_flag=True


    stage2_name = 'inferTs_CL'
    test_pred_name = 'nnCL'


    for i in range(1):
        fold = i+1
        save_path_show = 'New_Show/NNLVP_fold%d/show' % fold
        # 预测结果已经得到，建议全部先转化为numpy 再做处理！！


        #nii_testpreddir = 'F:/New_LVP/nn_S2_140_v3/%d/test/%s'%(fold,stage2_name)
        nii_testpreddir ='../Data/nii/S2/%s' % (stage2_name)

        # np_testpreddir = 'F:/New_LVP/nn_S2_140_v3/%d/test/%s_np'%(fold,stage2_name)
        np_testpreddir ='../Data/np/test/S2_pred2D'

        #第一阶段的预测结果
        S1np_testpreddir = '../Data/np/test/S1_pred2D'
        testimagedir = '../Data/np/test/image3D'
        testlabeldir = '../Data/np/test/label3D'
        testpreddir = '../Data/np/test/NNLVP_pred3D'
        show_path = '../Data/np/test/NNLVP_show'
        mk_dir(testpreddir)

        niipred_2np(nii_testpreddir, np_testpreddir)


        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        # 以三维为单位

        test_dataset = BasicDataset_3D(testimagedir, testlabeldir)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=opt.valbatchsize,
                                                      num_workers=opt.workers)

        # 由于需要测量豪斯多夫距离，所以补充
        normal_test(test_dataloader, [256, 128], itr=5, fold_id=fold,
                    np_testpreddir=np_testpreddir, S1_np_testpreddir=S1np_testpreddir, cav_first_flag=False,testpreddir=testpreddir,show=show_path)





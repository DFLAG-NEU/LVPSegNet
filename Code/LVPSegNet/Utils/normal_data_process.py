
import sys
from skimage import measure
import shutil
import os
import numpy as np
# from Utils.show import combine_imglabel,image,combine_imglabel_dye
from collections import OrderedDict
from data_aug import data_augment
import json
import pydicom
from PIL import Image
import SimpleITK as sitk
import random
import math
def normalization(input):
    return (input-np.min(input))/(np.max(input)-np.min(input))

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
#获取对应关系信息
def bmp2np(input,modal='PSIR'):
    lb=input
    unique_value = np.unique(np.array(lb))
    palette = np.array(lb.getpalette(), dtype=np.uint8).reshape((256, 3))
    postmapped_value = []
    if modal == 'PSIR':
        for m in unique_value:
            psv = palette[m]
            if psv[0] == 255 and psv[1] == 0 and psv[2] == 0:
                postmapped_value.append(1)
            elif psv[0] == 0 and psv[1] == 255 and psv[2] == 0:
                postmapped_value.append(2)
            elif psv[0] == 0 and psv[1] == 0 and psv[2] == 255:
                postmapped_value.append(3)
            elif psv[0] == 0 and psv[1] == 0 and psv[2] == 0:
                postmapped_value.append(0)
            elif psv[0] == 255 and psv[1] == 0 and psv[2] == 255:
                print('----------------------')
                postmapped_value.append(1)
            elif psv[0] == 255 and psv[1] == 255 and psv[2] == 0:
                postmapped_value.append(4)
            else:
                print('wrong label value', end='')
                print(psv)

    lb = np.array(lb)
    true_lb = np.zeros_like(lb)
    for m in range(len(unique_value)):
        true_lb[np.where(lb == unique_value[m])] = postmapped_value[m]  # blue
    lb = true_lb
    return lb


# def make_voxel_map():
#
#
#
#     with open('G:/Task_M7D27/Utils/map_info/dcm_map_patientindex_info_1.json' , "r", encoding="utf-8") as f:
#         content_all = json.load(f)
#     with open('G:/Task_M7D27/Utils/map_info/dcm_map_patientindex_info_2.json', "r", encoding="utf-8") as f:
#         content = json.load(f)
#     content_all.update(content)
#     with open('G:/Task_M7D27/Utils/map_info/dcm_map_patientindex_info_3.json', "r", encoding="utf-8") as f:
#         content = json.load(f)
#     content_all.update(content)
#     with open('G:/Task_M7D27/Utils/map_info/dcm_map_patientindex_info_4.json', "r", encoding="utf-8") as f:
#         content = json.load(f)
#
#
#     content_all.update(content)
#     json_dict = OrderedDict()
#
#     for i in content_all:
#         dcm_list=content_all[i]
#         dcm_name=dcm_list[1]
#
#         dcm_tag_1 = sitk.ReadImage(dcm_name)
#         print(dcm_tag_1.GetSpacing())
#         spacex,spacey,spacez=dcm_tag_1.GetSpacing()
#
#         # dcm_tag_1 = pydicom.read_file(dcm_name)
#         # # 获取像素间距.
#         # spacex, spacey = dcm_tag_1.PixelSpacing
#
#         if not spacex<1:
#             print(i)#需要调整体素的大小
#             print(spacex)
#             print(spacey)
#
#             spacex=0.89285713434219
#             spacey =0.89285713434219
#             #后期自己进行调整
#         #json_dict[i]=np.array([spacex,spacey,spacez])
#         json_dict[i]=[spacex,spacey,spacez]
#
#     with open('voxel_size.json', 'w') as f:
#         json.dump(json_dict, f, indent=4, sort_keys=True)
#
#
#     #生成对应的voxel map

def voxel_map_update():
    #生成对应体素大小都是多少的map
    with open('voxel_size.json' , "r", encoding="utf-8") as f:
        content_all = json.load(f)
    raw_path='F:/IMH_Data_7'
    with open('../Y24_M1D4_S2/map_info/index_map_patient_info.json', "r", encoding="utf-8") as f:
        content_p= json.load(f)

    json_dict = OrderedDict()
    pest=os.listdir(raw_path)
    for p in pest:
        if p in content_p:
            if int(content_p[p])>=80:
                dcm_path = raw_path + '/' + p + '/PSIR'
                fset = os.listdir(dcm_path)
                dcm_name = ''
                for f in fset:
                    if f.endswith('.dcm'):
                        dcm_name = f
                        break
                dcm_name = dcm_path + '/' + dcm_name
                dcm_tag_1 = sitk.ReadImage(dcm_name)
                print(dcm_tag_1.GetSpacing())
                spacex, spacey, spacez = dcm_tag_1.GetSpacing()

                if not spacex < 1:
                    print(p)  # 需要调整体素的大小
                    print(spacex)
                    print(spacey)

                    spacex = 0.89285713434219
                    spacey = 0.89285713434219
                    # 后期自己进行调整
                # json_dict[i]=np.array([spacex,spacey,spacez])
                content_all[content_p[p]] = [spacex, spacey, spacez]

    with open('voxel_size_new.json', 'w') as f:
        json.dump(content_all, f, indent=4, sort_keys=True)


    #生成对应的voxel map

def data3D_to_2D(path='../Data/np',cnt=0,need_data_augment=False,mode='test'):#M12D1用于交叉验证的时候

    #同时找到原始的DCM文件，尽量的详细

    #new_label_path='G:/M7D25/wholeset_modified/'

    # json_dict = OrderedDict()
    # json_dict_2D = OrderedDict()
    # json_dict_index = OrderedDict()
    #
    # with open('G:/Task_M4D25/utils/map_info/dcm_map_patientindex_info_%d.json' % 1, "r", encoding="utf-8") as f:
    #     content_all= json.load(f)
    # with open('G:/Task_M4D25/utils/map_info/dcm_map_patientindex_info_%d.json' % 2, "r", encoding="utf-8") as f:
    #     content = json.load(f)
    # content_all.update(content)
    # with open('G:/Task_M4D25/utils/map_info/dcm_map_patientindex_info_%d.json' % 3, "r", encoding="utf-8") as f:
    #     content = json.load(f)
    # content_all.update(content)
    # with open('G:/Task_M4D25/utils/map_info/dcm_map_patientindex_info_%d.json' % 4, "r", encoding="utf-8") as f:
    #     content = json.load(f)
    # content_all.update(content)
    # content=content_all

    pset=os.listdir(path+'/image3D')
    make_dir(path+'/image')
    make_dir(path + '/label')
    #找到对应的dcm文件
    #index
    for p in pset:
        image=np.load(path+'/image3D/'+p)
        label=np.load(path+'/label3D/'+p)
        #比如'0'
        p_index=p[:-4]
        # dcm_list=content[p_index]

        slice_index_list=[]
        slice_dcm_list=[]

        for i in range(image.shape[2]):
            slice_image=image[:,:,i].copy()
            slice_label = label[:, :, i].copy()


            np.save(path+'/image/%d.npy'%cnt,slice_image)
            np.save(path+'/label/%d.npy'%cnt,slice_label)

            # slice_index_list.append(cnt)
            #
            # d=dcm_list[i]
            # buf = d
            # sub = "/"
            # index = [substr.start() for substr in re.finditer(sub, buf)]
            # file_name = (buf[index[3] + 1:])[:-4] + '.bmp'
            # slice_dcm_list.append(new_label_path+file_name)
            # json_dict_2D[cnt]=new_label_path+file_name

            cnt+=1
            #存储每张的切片

        #以三维为角度

        #设置好对照的顺序
        # json_dict[p_index]=slice_dcm_list
        # json_dict_index[p_index]=slice_index_list

        #是否需要进行数据增广操作？？？

    # with open('map_info/patient_map_sliceindex_info_%d_%s.json' % (fold,mode), 'w') as f:
    #     json.dump(json_dict_index, f, indent=4, sort_keys=True)



    # with open('map_info/dcm_map_sliceindex_info_%d_%s.json' % (fold,mode), 'w') as f:
    #     json.dump(json_dict_2D, f, indent=4, sort_keys=True)
    #
    # if mode=='test':
    #     with open('map_info/dcm_map_patientindex_info_%d.json' % (fold), 'w') as f:
    #         json.dump(json_dict, f, indent=4, sort_keys=True)
    #     with open('map_info/patient_map_sliceindex_info_%d_%s.json' % (fold,mode), 'w') as f:
    #         json.dump(json_dict_index, f, indent=4, sort_keys=True)


    #cnt=563
    if need_data_augment:#进行数据增广
        make_dir(path + '/auged_label')
        make_dir(path + '/auged_image')
        #拷贝

        for i in range(cnt):
            shutil.copy(path  + '/image/' + str(i) + '.npy',path  + '/auged_image/' + str(i) + '.npy')
            shutil.copy(path  + '/label/' + str(i) + '.npy',path  + '/auged_label/' + str(i) + '.npy')
        data_augment(path=path,cnt=cnt)
        #同时将raw_image label 也放入 auged数据集当中去
    # with open('map_info/patient_map_sliceindex_info_%d_%sauged.json' % (fold,mode), 'w') as f:
    #     json.dump(json_dict_index, f, indent=4, sort_keys=True)
    # if show_flag:
    #     #进行数据集展示
    #     try:
    #         os.makedirs(path + '/show')
    #     except OSError:
    #         pass
    #     iset=os.listdir(path+'/image')
    #     for i in iset:
    #         image=np.load(path+'/image/'+i)
    #         label = np.load(path + '/label/' + i)
    #         out=combine_imglabel_dye(image.copy(),label)
    #         out.save(path + '/show/'+i[:-4]+'.png')
    #     if need_data_augment:
    #         iset = os.listdir(path + '/auged_image')
    #         for i in iset:
    #             image = np.load(path + '/auged_image/' + i)
    #             label = np.load(path + '/auged_label/' + i)
    #             out = combine_imglabel_dye(image.copy(), label)
    #             out.save(path + '/show/' + i[:-4] + '.png')

def dataraw_to_3D(save_path='../Data/np'):

    path='../Data/Raw'
    make_dir(save_path+'/label3D')
    make_dir(save_path + '/image3D')

    #事先没有保证体素大小的统一
    pset=os.listdir(path)
    for p in pset:

        fset=os.listdir(path+'/'+p+'/PSIR')
        order_list = []

        for f in fset:
            if f.endswith('.dcm'):
                id = int(f[:-4])
                order_list.append(id)

                ds = pydicom.read_file(path+'/'+p +'/PSIR'+ '/%s' % f)
                pa = ds.pixel_array


        order_list = np.array(order_list)
        sortedindex = np.sort(order_list)

        image_3D=np.zeros((pa.shape[0],pa.shape[1],len(order_list)))
        label_3D=np.zeros((pa.shape[0],pa.shape[1],len(order_list)))

        for f in fset:
            if f.endswith('.dcm'):
                id = int(f[:-4])

                ds = pydicom.read_file(path+'/'+p +'/PSIR'+ '/%s' % f)
                pa = ds.pixel_array

                print(pa.shape)
                print(ds[0x28, 0x30].value)

                if not os.path.exists(path+'/'+p +'/PSIR'+ '/%s.bmp' % (f[:-4])):  # 标签图像未对应
                    print('image and label do not match in %s' % p)
                    continue

                lb = Image.open(path+'/'+p +'/PSIR'+ '/%s.bmp' % (f[:-4]))
                slice_label=bmp2np(input=lb)
                slice_image=pa
                slice_image=normalization(slice_image)
                label_3D[:, :, (np.where(sortedindex == id))[0][0]] = slice_label
                image_3D[:, :, (np.where(sortedindex == id))[0][0]] = slice_image


        label_3D=label_3D.astype(np.uint8)
        np.save(save_path+'/label3D/%s.npy'%p,label_3D)
        np.save(save_path + '/image3D/%s.npy' % p, image_3D)

def dataset_split():
    print('')
    #数据集划分60  7:1:2 42:6:12
    srcpath='../Data/np/image3D'
    srcpath_label = '../Data/np/label3D'

    pset=os.listdir(srcpath)
    total_num = len(pset)

    cv_num = (total_num-math.ceil(0.1*total_num)-math.ceil(0.2*total_num), math.ceil(0.1*total_num), math.ceil(0.2*total_num))
    train_set=random.sample(pset,cv_num[0]+cv_num[1])
    test_set=[]
    for t in pset:
        if not t in train_set:
            test_set.append(t)
    pure_train_set=[]
    val_set=random.sample(train_set,cv_num[1])
    for t in train_set:
        if not t in val_set:
            pure_train_set.append(t)
    #进行划分
    make_dir('../Data/np/train/image3D')
    make_dir('../Data/np/test/image3D')
    make_dir('../Data/np/val/image3D')

    make_dir('../Data/np/train/label3D')
    make_dir('../Data/np/test/label3D')
    make_dir('../Data/np/val/label3D')

    for p in pset:
        if p in pure_train_set:
            shutil.copy(srcpath+'/%s'%p,'../Data/np/train/image3D/%s'%p)
            shutil.copy(srcpath_label+'/%s'%p,'../Data/np/train/label3D/%s'%p)
        elif p in test_set:
            shutil.copy(srcpath+'/%s'%p,'../Data/np/test/image3D/%s'%p)
            shutil.copy(srcpath_label+'/%s'%p,'../Data/np/test/label3D/%s'%p)
        else:
            shutil.copy(srcpath+'/%s'%p,'../Data/np/val/image3D/%s'%p)
            shutil.copy(srcpath_label+'/%s'%p,'../Data/np/val/label3D/%s'%p)

    #分别进行保存，数据划分形式
if __name__=="__main__":
    dataraw_to_3D()
    dataset_split()
    data3D_to_2D(path='../Data/np/train',need_data_augment=True)
    data3D_to_2D(path='../Data/np/test', need_data_augment=False)
    data3D_to_2D(path='../Data/np/val', need_data_augment=False)

    sys.exit()













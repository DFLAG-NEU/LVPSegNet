import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import pydicom
import nibabel as nib
from collections import OrderedDict
import json
from dicom2nifti import dicom_series_to_nifti
import pandas as pd
from scipy.ndimage.interpolation import zoom
import shutil

from PIL import Image
import re
def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key



def interpolate(input,scale,method):
    return zoom(input, (scale,scale), order=method)

def normalization(input,value_max,value_min):
    return (input-value_min)/(value_max-value_min)

def dcm2nii(path_read, path_save):
    dicom_series_to_nifti(path_read,path_save)


def dcm2nii_new(path_read, path_save):  # from CSDN;function: transfer dcm_series into nii file
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    # print(len(series_file_names))  #11
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)


def _get_instance_number(dicom_path):
    img_reader = sitk.ImageFileReader()
    img_reader.SetFileName(dicom_path)
    img_reader.LoadPrivateTagsOn()
    img_reader.ReadImageInformation()
    number_str = img_reader.GetMetaData('0020|0013')  # 获取Instance number
    return int(number_str)


def get_slice(dcm_path):
    # 用这个函数获得按照Instance number排序的切片路径
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)  # 获得切片路径，这个是按切片命名排序的
    #dicom_names=('C:/Users/admin/Desktop/PSIR/2.dcm', 'C:/Users/admin/Desktop/PSIR/3.dcm', 'C:/Users/admin/Desktop/PSIR/4.dcm', 'C:/Users/admin/Desktop/PSIR/5.dcm','C:/Users/admin/Desktop/PSIR/6.dcm', 'C:/Users/admin/Desktop/PSIR/7.dcm')
    r = []
    for name in dicom_names:
        r.append({"instance_number": _get_instance_number(name), "dcm_name": name})
    r = pd.DataFrame(r)
    r = r.sort_values("instance_number")  # 按照Instance number排序
    r = tuple(r["dcm_name"])  # 获得按照Instance number排序的切片路径
    return r

def dcm2nii_new2(path_read, path_save):
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = get_slice(path_read)

    #在这里进行一些修改，生成我们的dicom_names
    dicom_names=[]
    fset=os.listdir(path_read)
    order_list=[]
    name_list=[]
    for f in fset:
        if f.endswith('.dcm'):#依次加入到列表当中
            id = f[:-4]
            id = int(id)
            order_list.append(id)
            name_list.append(path_read+'/'+f)
    order_list = np.array(order_list)
    sortedindex = np.argsort(order_list)
    for i in sortedindex:
        dicom_names.append(name_list[i])
    dicom_names=tuple(dicom_names)
    reader.SetFileNames(dicom_names)
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    origin = image.GetOrigin()  # x, y, z
    #     print(origin)
    spacing = image.GetSpacing()  # x, y, z
    #     print(spacing)
    direction = image.GetDirection()  # x, y, z
    #     print(direction)
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)  # 这些需要设置一下，不然z轴的空间信息可能会被压缩
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3,path_save)

def mk_nii_label(readpath,labelpath,savepath):#生成nii文件的label
    img=nib.load(readpath)
    label=np.load(labelpath)
    if (img.dataobj).shape[0]==256:


        #直接读取
        buf = labelpath
        sub = "/"
        index = [substr.start() for substr in re.finditer(sub, buf)]
        file_name=buf[index[-1]+1:]
        true_label=np.load('G:/Task_M7D27/Data/Data_256/'+file_name)
        label=true_label
    label = np.swapaxes(label, 0, 1)
    assert (img.dataobj).shape[2] == label.shape[2]
    #
    label_nifti = nib.Nifti1Image(label, img.affine,img.header)
    nib.save(label_nifti, savepath)

    #还涉及到方向变反了的操作

def mk_nii_label_S1(readpath,labelpath,savepath):#生成nii文件的label
    img=nib.load(readpath)
    label=np.load(labelpath)

    # if (img.dataobj).shape[0]==256:
    #     #此时需要对标签集进行进一步处理
    #
    #     buf = labelpath
    #     sub = "/"
    #     index = [substr.start() for substr in re.finditer(sub, buf)]
    #     file_name=buf[index[-1]+1:]
    #     true_label=np.load('G:/Task_M7D27/Data/Data_256/'+file_name)
    #     label=true_label

    label=np.where(label==3,1,label)
    label=np.where(label == 2, 1, label)
    label=np.where(label==4,2,label)#转换成0,1,2
    label = np.swapaxes(label, 0, 1)
    assert (img.dataobj).shape[2] == label.shape[2]
    #
    label_nifti = nib.Nifti1Image(label, img.affine,img.header)
    nib.save(label_nifti, savepath)

    #还涉及到方向变反了的操作



def mk_nii_interpolation(readpath,labelpath,savepath):#生成nii文件的插值版本！！
    img=nib.load(readpath)
    label=np.load(labelpath)
    assert (img.dataobj).shape[2] == label.shape[2]
    label_nifti=nib.Nifti1Image(label, img.affine)
    nib.save(label_nifti, savepath)



def dcm2nii_sitk(path_read, path_save):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        lens[i] = len(dicom_names)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    sitk.WriteImage(image, path_save+'/data.nii.gz')

def np2nii_sitk(img, path_save,affine=None):#直接将numpy转为nii
    if affine==None:
        nib.save(nib.Nifti1Image(img,np.eye(4)),path_save)
    #sitk.WriteImage(image, path_save+'/data.nii.gz')
def dataset_dcm2nii_sitk(path_read, path_save):
    #以数据集为单位

    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        lens[i] = len(dicom_names)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    sitk.WriteImage(image, path_save+'/data.nii.gz')


def convert_dataset_2nii_4nnUnet_4stage1():

    #为nnUNet格式化数据集做准备,同时为我们方法的第一阶段做准备
    #将整个数据集设置为nii模式！
    ####将标签转化为nifti形式！需要与img相对应,同时文件名称要与我们目前的实验相对应！！
    show=True
    # with open('G:/Task_M4D25/utils/map_info/dcmlist_map_patientindex.json' , "r", encoding="utf-8") as f:
    #     content = json.load(f)
    # voxel_list = np.load('voxel_size_list.npy')

    pathimg='../Data/Raw'
    s=os.listdir(pathimg)
    p_name_list=[]
    cnt=0
    #show
    for p in s:
        p_name_list.append(p)
        dcmpath = pathimg + '/' + p + '/PSIR'
        # 此处不需要再生成image类了

        true_index = int(p)

        dcm2nii_new2(dcmpath, path_save='../Data/nii/whole_set/%d.nii.gz' % true_index)  # 具体什么index需要斟酌
        # cnt即为对应的index

        # 检查图像尺寸！！256,256需要额外处理
        ###进一步保存对应的标签文件！
        mk_nii_label_S1(readpath='../Data/nii/whole_set/%d.nii.gz' % true_index,
                        labelpath='../Data/np/label3D/%d.npy' % true_index,  # 我们后来自己构造的四类标签
                        savepath='../Data/nii/whole_set/%d_label_S1.nii.gz' % true_index
                        )
        # 将label文件也用作nifti存储
        cnt += 1
        ###甚至可能需要涉及到插值操作！！


    # if show:#对生成的结果进行展示，以此slice
    #     for j in range(80):
    #         #找到j所处的位置
    #         print('---------------------')
    #         # print(p_name_list[pre_j])
    #         # print(voxel_list[pre_j])
    #         #依次输出体素信息
    #         tmp=nib.load('../Data/nii_data/%d.nii.gz'%j)
    #         print(tmp.header['pixdim'])#牛的 看来是没有什么问题！
    #         tmp=nib.load('../Data/nii_data/%d_label.nii.gz'%j)
    #         print(tmp.header['pixdim'])#牛的 看来是没有什么问题！

            #nii3D_label_show(readpath='F:/Ours_nii_data/%d.nii.gz'%j,labelpath='F:/Ours_nii_data/%d_label_S1.nii.gz'%j,savepath='F:/Ours_nii_data_show_S1/%s_%d'%(p_name_list[pre_j],j))

def nn_CV_make_S1(save_path='../Data/nii/nnUNet_S1'):
    make_dir(save_path+'/nnUNet_preprocessed')
    make_dir(save_path + '/nnUNet_raw_data_base/nnUNet_raw_data')
    make_dir(save_path + '/nnUNet_raw_data_base/nnUNet_raw_data/Task001_MI/imagesTs')
    make_dir(save_path + '/nnUNet_raw_data_base/nnUNet_raw_data/Task001_MI/imagesTr')
    make_dir(save_path + '/nnUNet_raw_data_base/nnUNet_raw_data/Task001_MI/labelsTr')
    make_dir(save_path + '/nnUNet_raw_data_base/nnUNet_raw_data')
    make_dir(save_path + '/nnUNet_trained_models')

    save_path=save_path + '/nnUNet_raw_data_base/nnUNet_raw_data'
    nii_path='../Data/nii/whole_set'
    np_path='../Data/np'

    train_cases = []
    test_cases = []
    pset_test = os.listdir(np_path  + '/test/image3D')
    pset_train = os.listdir(np_path  + '/train/image3D')
    pset_val = os.listdir(np_path + '/val/image3D')
    # 生成测试集
    for p in pset_test:
        index = p[:-4]

        imgpath = nii_path + '/%s.nii.gz' % index
        labelpath = nii_path + '/%s_label_S1.nii.gz' % index

        index = int(index)

        index = index / 1000
        index = str(index)
        index = index[2:]
        while len(index) < 3:
            index += '0'
        test_cases.append(index)
        name = 'case_' + index + '_0000.nii.gz'
        # 即为我们要保存的
        shutil.copy(imgpath, save_path + '/Task001'  + '_MI'  + '/imagesTs/' + name)
        # shutil.copy(labelpath)
        # 无需保存测试集的label
    ###生成训练集

    for p in (pset_train+pset_val):
        index = p[:-4]

        imgpath = nii_path + '/%s.nii.gz' % index
        labelpath = nii_path + '/%s_label_S1.nii.gz' % index

        index = int(index)
        index = index / 1000
        index = str(index)
        index = index[2:]
        while len(index) < 3:
            index += '0'
        train_cases.append(index)
        name = 'case_' + index + '_0000.nii.gz'
        label_name = 'case_' + index + '.nii.gz'
        shutil.copy(imgpath, save_path + '/Task001' + '_MI' + '/imagesTr/' + name)
        shutil.copy(labelpath, save_path + '/Task001' + '_MI' + '/labelsTr/' + label_name)



    json_dict = OrderedDict()
    json_dict['name'] = "MVO"
    json_dict['description'] = "MI and Cav segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "MI data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MR",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "myocardium",
        "2": "Cavity"
    }
    json_dict['numTraining'] = len(train_cases)  # 60+20
    json_dict['numTest'] = len(test_cases)

    json_dict['training'] = [{'image': "./imagesTr/case_%s.nii.gz" % i, "label": "./labelsTr/case_%s.nii.gz" % i} for i
                             in
                             train_cases]
    json_dict['test'] = ["./imagesTs/case_%s.nii.gz" % i for i in test_cases]  ##有测试集就解开注释

    with open(os.path.join(save_path + '/Task001' + '_MI', "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
# read a dicom file
if __name__=="__main__":
    convert_dataset_2nii_4nnUnet_4stage1()
    nn_CV_make_S1()








import os
import numpy as np
import random
from scipy.ndimage import rotate
from ED import elastic_transform
def normalization(input):
    return (input-np.min(input))/(np.max(input)-np.min(input))

def find_key(value,dict):
    for d in dict:
        l=dict[d]
        if value in l:
            return d

def data_augment(path,cnt=0,rotate_num=2,ED_num=2,dict=None):


    if dict is None:
        iset = os.listdir(path + '/image')
        for i in iset:
            image = np.load(path + '/image/' + i)
            label = np.load(path + '/label/' + i)
            # rotate
            count = 0
            while count < rotate_num:  # 进行多少次弹性形变
                r_angle = random.randint(1, 359)
                rotated_image = rotate(image.copy(), angle=r_angle, reshape=False)
                rotated_label = rotate(label.copy(), angle=r_angle, order=0, reshape=False)
                # 如果处理出了超出范围的值，再进行归一化
                if (rotated_image < 0).any() or (rotated_image > 1).any():
                    rotated_image = normalization(rotated_image)
                # 如果存在超出范围的情况，则再进行一次归一化
                assert np.max(rotated_label) <= 4
                assert np.min(rotated_label) >= 0

                np.save(path + '/auged_image/' + str(cnt) + '.npy', rotated_image)
                np.save(path + '/auged_label/' + str(cnt) + '.npy', rotated_label)
                cnt += 1
                ####flip
                count += 1
            # FLIP进行一次就足够了
            c = random.randint(0, 2)  # 共3种情况
            if c == 0:
                f_image = np.flipud(image).copy()
                f_label = np.flipud(label).copy()
            elif c == 1:
                f_image = np.fliplr(image).copy()
                f_label = np.fliplr(label).copy()
            elif c == 2:
                f_image = np.flipud(image).copy()
                f_label = np.flipud(label).copy()
                f_image = np.fliplr(f_image).copy()
                f_label = np.fliplr(f_label).copy()
            np.save(path + '/auged_image/' + str(cnt) + '.npy', f_image)
            np.save(path + '/auged_label/' + str(cnt) + '.npy', f_label)
            cnt += 1
            # ED
            count = 0
            while count < ED_num:  # 进行多少次弹性形变
                image_t, label_t = elastic_transform(image, label, image.shape[1] * 2, image.shape[1] * 0.08,
                                                     image.shape[1] * 0.08)  # 参数值设定
                if (image_t < 0).any() or (image_t > 1).any():
                    image_t = normalization(image_t)
                # 如果存在超出范围的情况，则再进行一次归一化
                assert np.max(label_t) <= 4
                assert np.min(label_t) >= 0
                np.save(path + '/auged_image/' + str(cnt) + '.npy', image_t)
                np.save(path + '/auged_label/' + str(cnt) + '.npy', label_t)
                cnt += 1
                count += 1
    else:
        iset = os.listdir(path + '/image')
        for i in iset:
            p_index=find_key(value=int(i[:-4]), dict=dict)
            aug_list=[cnt,cnt+1,cnt+2,cnt+3,cnt+4]
            pre_list=dict[p_index]
            new_list=pre_list+aug_list
            dict[p_index]=new_list
            cnt=cnt+5
        return dict



import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
# Function to distort image  alpha = im_merge.shape[1]*2、sigma=im_merge.shape[1]*0.08、alpha_affine=sigma
#affine:仿射的
#alpha 作为
def elastic_transform(image,label ,alpha, sigma, alpha_affine, random_state=None):#参数值的设定还有待商榷
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """



    #对于0-1取值范围内的图像进行增广后，如果出现0-1的数据怎么办？？？？？
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size=shape   #(512,512)表示图像的尺寸
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
    # 其中center_square是图像的中心，square_size=512//3=170
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    #var1=random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)#随机值的取值范围是-sigma至sigma
    # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
    M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
    #默认使用 双线性插值，

    image = cv2.warpAffine(image, M, shape_size, borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, M, shape_size, flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
    #首先对图像进行随机仿射变换
    # # random_state.rand(*shape) 会产生一个和 shape 一样的服从[0,1]均匀分布的矩阵
    # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
    # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
    # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
    # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
    dx = gaussian_filter((random_state.uniform(-1,1,size=shape).astype(np.float32)), sigma) * alpha
    dy = gaussian_filter((random_state.uniform(-1,1,size=shape).astype(np.float32)), sigma) * alpha
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y= np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image,indices, order=1, mode='reflect').reshape(shape),map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

if __name__ == '__main__':

    img_path = '../image/0.npy'
    mask_path = '../label/0.npy'
    img=np.load(img_path)
    mask=np.load(mask_path)
    img_raw = img.copy()
    mask_raw=mask.copy()
    gt=merge(img_raw,mask_raw)
    # img=np.expand_dims(img, axis=2)
    # mask = np.expand_dims(mask, axis=2)
    # merge = np.concatenate((img,mask), axis=2)
    im_t,im_mask_t = elastic_transform(img,mask, img.shape[1] * 2, img.shape[1] * 0.08,
                                   img.shape[1] * 0.08)

    result=merge(im_t,im_mask_t)
    gt.show()
    result.show()
    im_t, im_mask_t = elastic_transform(img, mask, img.shape[1] * 2, img.shape[1] * 0.08,
                                        img.shape[1] * 0.08)

    result = merge(im_t, im_mask_t)
    result.show()



    #二者结合再输出一下

    #

    # img_path =  '/home/changzhang/ liubo_workspace/tmp_for_test/img'
    # mask_path = '/home/changzhang/liubo_workspace/tmp_for_test/mask'
    img_list = sorted(os.listdir(img_path))
    mask_list = sorted(os.listdir(mask_path))
    print(img_list)
    img_num = len(img_list)
    mask_num = len(mask_list)
    assert img_num == mask_num, 'img number is not equal to mask num.'
    count_total = 0
    for i in range(img_num):
        print(os.path.join(img_path, img_list[i]))   #将路径和文件名合成一个整体
        im = cv2.imread(os.path.join(img_path, img_list[i]), -1)
        im_mask = cv2.imread(os.path.join(mask_path, mask_list[i]), -1)
        # # Draw grid lines
        # draw_grid(im, 50)
        # draw_grid(im_mask, 50)
        # Merge images into separete channels (shape will be (cols, rols, 2))!!

        im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
        # get img and mask shortname
        (img_shotname, img_extension) = os.path.splitext(img_list[i])  #将文件名和扩展名分开
        (mask_shotname, mask_extension) = os.path.splitext(mask_list[i])
        # Elastic deformation 10 times 一共进行十次弹性形变，达到增加数据量的目的
        count = 0
        while count < 10:
            # Apply transformation on image  im_merge.shape[1]表示图像中像素点的个数
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                           im_merge.shape[1] * 0.08)

            # Split image and mask
            im_t = im_merge_t[..., 0]
            im_mask_t = im_merge_t[..., 1]
            # save the new imgs and masks
            cv2.imwrite(os.path.join(img_path, img_shotname + '-' + str(count) + img_extension), im_t)
            cv2.imwrite(os.path.join(mask_path, mask_shotname + '-' + str(count) + mask_extension), im_mask_t)
            count += 1
            count_total += 1
        if count_total % 100 == 0:
            print('Elastic deformation generated {} imgs', format(count_total))
            # # Display result
            # print 'Display result'
            # plt.figure(figsize = (16,14))
            # plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
            # plt.show()
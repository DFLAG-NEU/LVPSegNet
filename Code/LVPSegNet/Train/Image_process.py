import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
def interpolate(input,scale,method):
    return zoom(input, (scale,scale), order=method)

def z_score_normalization(input):
    return (input-np.mean(input))/np.std(input)
def norm(input):
    if type(input) is np.ndarray:
        input=(input-np.min(input))/(np.max(input)-np.min(input))
    else:
        input=(input-torch.min(input))/(torch.max(input)-torch.min(input))
    return input
def normalization(input):
    return (input-np.min(input))/(np.max(input)-np.min(input))
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
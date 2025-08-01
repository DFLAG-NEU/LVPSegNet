from torch.utils.data import Sampler
import random
import sys
import numpy as np
import math
class CLSampler(Sampler):
    def __init__(self, batch_size, diff_list,valid_num,drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.diff_list=diff_list
        self.valid_num=valid_num
        #同时计算一下batch indice吧

    def __iter__(self):
        valid_order = []
        non_valid_order = []

        valid_list = self.diff_list.copy()[:self.valid_num]
        non_valid_list = self.diff_list.copy()[self.valid_num:]

        random.shuffle(valid_list)
        if not len(non_valid_list) == 0:
            random.shuffle(non_valid_list)

        tmp = []
        for m in range(len(valid_list)):
            tmp.append(int(valid_list[m]))
            if (m + 1) % self.batch_size == 0 or (m + 1) == len(valid_list):
                valid_order.append(tmp)
                tmp = []

        if not len(non_valid_list) == 0:
            tmp = []
            for m in range(len(non_valid_list)):
                tmp.append(int(non_valid_list[m]))
                if (m + 1) % self.batch_size == 0 or (m + 1) == len(non_valid_list):
                    non_valid_order.append(tmp)
                    tmp = []

        groups=valid_order+non_valid_order

        # random.shuffle(self.groups)
        for group in groups:
            yield group

    def __len__(self):
        if not self.drop_last:
            return (math.ceil(self.valid_num/ self.batch_size))+(math.ceil((len(self.diff_list)-self.valid_num)/ self.batch_size))
        else:
            print('false!!!')
            sys.exit()

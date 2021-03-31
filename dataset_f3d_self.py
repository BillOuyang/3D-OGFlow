"""
Flyingthings3D dataloader for the self-supervised training,

"""

import glob
import numpy as np
import os
import time
import random

import torch
from torch.utils.data import Dataset, DataLoader



class SceneflowDataset_self(Dataset):
    def __init__(self, npoints=4096, root='datasets/data_processed_maxcut_35_20k_2k_8192', train=True, cache=None):
        self.npoints = npoints
        self.train = train
        self.root = root

        self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))

        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32') / 255
                color2 = data['color2'].astype('float32') / 255
                flow = data['flow'].astype('float32')
                # mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow)

        # number of occlusions
        k_group = 6
        k_neighbor = 350

        # magnitude of the random translation
        flow_mag = 2.0
        random_flow = (torch.rand([1,3], dtype=torch.float32)-0.5) * flow_mag
        flow_self = random_flow.repeat([self.npoints, 1])


        n1 = pos1.shape[0]
        sample_idx_self = np.random.choice(n1, self.npoints + k_neighbor*k_group, replace=False)
        n2 = pos2.shape[0]
        sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

        pos1 = torch.from_numpy(pos1[sample_idx_self, :])
        color1 = torch.from_numpy(color1[sample_idx_self, :])
        flow = torch.from_numpy(flow[sample_idx_self, :])

        occ_mask = torch.ones(self.npoints + k_neighbor*k_group, dtype=torch.bool)

        for i in range(k_group):
            sq_dist = (pos1 - pos1[i]).norm(dim=-1)
            knn = sq_dist.topk(k=k_neighbor, largest=False)[1]
            occ_mask[knn] = False


        pos2_self = pos1[occ_mask, :]
        color2_self = color1[occ_mask, :]
        sample_idx2_self = np.random.choice(pos2_self.shape[0], self.npoints, replace=False)
        pos2_self = pos2_self[sample_idx2_self, :] + random_flow
        color2_self = color2_self[sample_idx2_self, :]


        sample_idx1 = np.random.choice(self.npoints + k_neighbor*k_group, self.npoints, replace=False)
        pos1 = pos1[sample_idx1, :]
        color1 = color1[sample_idx1, :]
        flow = flow[sample_idx1, :]
        mask1 = occ_mask[sample_idx1].unsqueeze(-1).type(torch.float32)


        pos2 = torch.from_numpy(pos2[sample_idx2, :])
        color2 = torch.from_numpy(color2[sample_idx2, :])


        pos1_center = torch.mean(pos1, 0)
        pos1 -= pos1_center
        pos2_self -= pos1_center
        pos2 -= pos1_center



        return pos1, pos2, color1, color2, flow, pos2_self, color2_self,flow_self, mask1

    def __len__(self):
        return len(self.datapath)



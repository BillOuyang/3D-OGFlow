"""
PointConv util functions
Author: Wenxuan Wu
Date: May 2020

Modified by BillOuyang
Date: March 2021

"""

import torch
import torch.nn as nn

import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2 import pointnet2_utils

LEAKY_RATE = 0.1
use_bn = False


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points


def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)  ## [B, s, nsample]
    grouped_xyz = index_points_group(s_xyz, idx)  # [B, s=npoint, nsample, C=3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)  ## [B, s, nsample,D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, N,
                                                                                                                -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points


class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points features, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)  ## [B, npoint]
        new_xyz = index_points_gather(xyz, fps_idx)  ## [B, npoint, c]

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        ##[b,s,nsample,c+d]   ## [B, s, nsample, c=3]

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)  ## ## [B, c, s, nsample]
        weights = self.weightnet(grouped_xyz)  ##[b,weightnet, nsample, s]
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                -1)  ##[b,s,weightnet*(c+d)]
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx


class Occ_weighted_CV(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp_occ, bottom_level=False, bn=use_bn, use_leaky=True):
        super(Occ_weighted_CV, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        # occlusion
        self.mlp_convs_occ = nn.ModuleList()
        last_channel = in_channel if bottom_level else in_channel+1
        for out_channel in mlp_occ:
            self.mlp_convs_occ.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

        self.fc_occ = nn.Sequential(
            nn.Conv1d(last_channel, 3, 1),
            nn.LeakyReLU(LEAKY_RATE, inplace=True),
            nn.Conv1d(3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, xyz1, xyz2, points1, points2, up_occ_mask=None):
        """
        Occlusion/Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
            up_occ_mask: input upsampled occlusion map, [B, 1, N1]
        Return:
            costVolume: Occlusion weighted Cost Volume, [B, D_cv, N1]
            occ_mask: predicted occlusion at current pyramid level, [B, 1, N1]
        """
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)  ## [B,N1, nsample, c=3]

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)

        # occ predictor
        if up_occ_mask is not None:
            new_points_occ = torch.cat([grouped_points1, grouped_points2, direction_xyz, up_occ_mask.permute(0, 2, 1).unsqueeze(3).repeat(1, 1, self.nsample,1)], dim=-1)
        else:
            new_points_occ = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)
        new_points_occ = new_points_occ.permute(0, 3, 2, 1) # [B, D1+D2+3+1, nsample, N1]
        for i, conv in enumerate(self.mlp_convs_occ):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points_occ = self.relu(bn(conv(new_points_occ)))
            else:
                new_points_occ = self.relu(conv(new_points_occ))

        new_points_occ = torch.max(new_points_occ, dim=2)[0]
        occ_mask = self.fc_occ(new_points_occ)  #[B, 1, N1]


        # C.V
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        # CV_cross
        costVoulume_cross = torch.max(new_points, 2)[0]

        # CV_self
        knn_idx_self = knn_point(self.nsample+32, xyz1, xyz1)
        costVolume_self = index_points_group(costVoulume_cross.permute(0, 2, 1), knn_idx_self)
        costVolume_self = torch.max(costVolume_self, 2)[0].permute(0, 2, 1)

        # Occ weighted CV
        costVolume = occ_mask * costVoulume_cross + (1.0 - occ_mask)*costVolume_self

        return costVolume, occ_mask


class Warping(nn.Module):

    def forward(self, xyz1, xyz2, flow1=None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1)  # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1)  # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C)  # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim=2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)  # B 3 N2

        return warped_xyz2


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        ## sparseflow/occlusion: [B, S, D]

        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1)  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1)  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1)  # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)  ## [B, N, nsample=3 3]
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)  ## [B, N, nsample=3]
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)  ## [B, N, 1]
        weight = (1.0 / dist) / norm  ## [B, N, 3]

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim=2).permute(0, 2, 1)
        return dense_flow


class SceneFlow_Estimator(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch=3, occ_ch=1,channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlow_Estimator, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch + occ_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, up_flow=None, occ_mask=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        mask: B 1 N
        '''
        if up_flow is None:
            new_points = torch.cat([feats, cost_volume, occ_mask], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, up_flow, occ_mask], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)

        if up_flow is not None:
            flow = up_flow + flow

        return new_points, flow.clamp(self.clamp[0], self.clamp[1])
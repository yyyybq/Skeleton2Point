'''
Date: 2022-03-11 11:01:07
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 10:01:25
'''

import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/data/yinbaiqiao/PointConT-master/pointnet2_ops_lib")
from pointnet2_ops import pointnet2_utils
#from pytorch3d.ops import sample_farthest_points

from pointnet_util import index_points, square_distance
from .ResMLP import ResMLPBlock1D


def Point2Patch(num_patches, patch_size, xyz):  # 512,16,[3, 3200, 3]
    """功能：在3D空间中对点云进行分块 构建点云数据的局部邻域
    Patch Partition in 3D Space
    Input:
        num_patches: number of patches, S
        patch_size: number of points per patch, k
        xyz: input points position data, [B, N, 3]
    Return:
        centroid: patch centroid, [B, S, 3]
        knn_idx: [B, S, k]
    """
    # FPS the patch centroid out
    fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), num_patches).long()  # [B, S]
    centroid_xyz = index_points(xyz, fps_idx)    # [B, S, 3] [3, 512, 3]
    # knn to group per patch
    dists = square_distance(centroid_xyz, xyz)  # [B, S, N] [3, 512, 3200]
    knn_idx = dists.argsort()[:, :, :patch_size]  # [B, S, k] [3, 512, 16]
    
    return centroid_xyz, fps_idx, knn_idx

def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class ContextCluster(nn.Module):
    def __init__(self, dim, heads=4, head_dim=24):
        super(ContextCluster, self).__init__()
        self.heads = heads
        self.head_dim=head_dim
        self.fc1 = nn.Linear(dim, heads*head_dim)
        self.fc2 = nn.Linear(heads*head_dim, dim)
        self.fc_v = nn.Linear(dim, heads*head_dim)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

    def forward(self, x): #[b,d,k]中间是通道
        #print(x.size())
        res = x
        x = rearrange(x, "b d k -> b k d")
        value = self.fc_v(x)  # [b,k,head*head_d]
        x = self.fc1(x) # [b,k,head*head_d]
        x = rearrange(x, "b k (h d) -> (b h) k d", h=self.heads)  # [b,k,d]
        value = rearrange(value, "b k (h d) -> (b h) k d", h=self.heads)  # [b,k,d]
        center = x.mean(dim=1, keepdim=True)  # [b,1,d]
        value_center = value.mean(dim=1, keepdim=True)  # [b,1,d]
        sim = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(center, x) )#[B,1,k]
        # out [b, 1, d]
        out = ( (value.unsqueeze(dim=1)*sim.unsqueeze(dim=-1) ).sum(dim=2) + value_center)/ (sim.sum(dim=-1,keepdim=True)+ 1.0) # [B,M,D]
        out = out*(sim.squeeze(dim=1).unsqueeze(dim=-1)) # [b,k,d]
        out = rearrange(out, "(b h) k d -> b k (h d)", h=self.heads)  # [b,k,d]
        out = self.fc2(out)
        out = rearrange(out, "b k d -> b d k")
        return res + out


class PatchAbstraction(nn.Module):
    def __init__(self, num_patches, patch_size, in_channel, mlp):
        super(PatchAbstraction, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_act = nn.ModuleList()
        self.mlp_res = ResMLPBlock1D(mlp[-1], mlp[-1])
        self.cocs = ContextCluster(mlp[-1])
        last_channel = in_channel 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))      # 6,64
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_act.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, xyz, feature):
        """
        Input: xyz [B, S_, 3]
               features [B, S_, C]
        Return: [B, S, 3+D]
        """
        B, _, C = feature.shape
        centroid_xyz, centroid_idx, knn_idx = Point2Patch(self.num_patches, self.patch_size, xyz)
        
        centroid_feature = index_points(feature, centroid_idx) # [B, S, C]
        grouped_feature = index_points(feature, knn_idx)    # [B, S, k, C]

        k = grouped_feature.shape[2]

        # Normalize                                                                                                                                                                                                                                                            
        grouped_norm = grouped_feature - centroid_feature.view(B, self.num_patches, 1, C) # [B, S, k, C]
        groups = torch.cat((centroid_feature.unsqueeze(2).expand(B, self.num_patches, k, C), grouped_norm), dim=-1) # [B, S, k, 2C]
        
        groups = groups.permute(0, 3, 2, 1) # [B, Channel, k, S]
        #groups = groups.__cuda_array_interface__()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            act = self.mlp_act[i]
            # torch.cuda.empty_cache()
            groups =  act(bn(conv(groups))) # [B, D, k, S]

        max_patches = torch.max(groups, 2)[0] # [B, D, S]
        max_patches = self.mlp_res(max_patches)#.transpose(1, 2) # [B, S, D]

        avg_patches = torch.mean(groups, 2)# [B, D, S]
        #global_patches = self.cocs(avg_patches).transpose(1, 2)#进的时候是[16,64,1024]
        max_patches = self.cocs(max_patches).transpose(1, 2)
        avg_patches = avg_patches.transpose(1, 2) # [B, S, D]
        return centroid_xyz, max_patches, avg_patches#, global_patches



class ConT(nn.Module):
    '''
    Content-based Transformer
    Args:
        dim (int): Number of input channels.
        local_size (int): The size of the local feature space.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''

    def __init__(self, dim, local_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., kmeans = False):

        super().__init__()
        self.dim = dim
        self.ls = local_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        '''
        Input: [B, S, D]
        Return: [B, S, D]
        '''

        B, S, D = x.shape # 3，1600，64
        nl = S // self.ls
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, h, S, d]

        q_pre = qkv[0].reshape(B*self.num_heads, S, D // self.num_heads).permute(0,2,1) # [B*h, d, S]
        ntimes = int(math.log(nl, 2))
        # q_idx_last = torch.arange(S).cuda().unsqueeze(0).expand(B*self.num_heads, S)
        q_idx_last = torch.arange(S).unsqueeze(0).expand(B*self.num_heads, S)

        
        # balanced binary clustering
        for _ in range(ntimes):
            bh,d,n = q_pre.shape # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.reshape(bh, d, 2, n//2) # [B*h*2^n, d, 2, S/2^n]
            q_avg = q_pre_new.mean(dim=-1) # [B*h*2^n, d, 2]

            q_avg = torch.nn.functional.normalize(q_avg.permute(0,2,1), dim=-1) 
            q_norm = torch.nn.functional.normalize(q_pre.permute(0,2,1), dim=-1)
  
            q_scores = square_distance(q_norm, q_avg) # [B*h*2^n, S/2^n, 2]
            q_ratio = (q_scores[:,:,0]+1) / (q_scores[:,:,1]+1) # [B*h*2^n, S/2^n]
            q_idx = q_ratio.argsort()
            q_idx_last = q_idx_last.cuda()
            q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh*2, n//2) # [B*h*2^n, S/2^n]
            q_idx_new = q_idx.unsqueeze(1).expand(q_pre.size()) # [B*h*2^n, d, S/2^n]
            q_pre = q_pre.cuda()
            q_pre_new = q_pre.gather(dim=-1, index=q_idx_new).reshape(bh, d, 2, n//2) # [B*h*2^n, d, 2, S/(2^(n+1))]
            q_pre = rearrange(q_pre_new, 'b d c n -> (b c) d n')   # [B*h*2^(n+1), d, S/(2^(n+1))]

        # clustering is performed independently in each head
        q_idx = q_idx_last.view(B,self.num_heads, S) # [B, h, S]
        q_idx_rev = q_idx.argsort() # [B, h, S]

        # cluster query, key, value 
        q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size()) # [3, B, h, S, d] d=16=d_org/h
        qkv = qkv.cuda()    # ljy add
        qkv_pre = qkv.gather(dim=-2, index=q_idx) # [3, B, h, S, d] [3, 3, 4, 1600, 16]
        q, k, v  = rearrange(qkv_pre, 'qkv b h (nl ls) d -> qkv (b nl) h ls d', ls=self.ls)

        # MSA
        attn = (q - k)*self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out =  torch.einsum('bhld, bhld->bhld', attn, v) # [B*(nl), h, ls, d]

        # merge and reverse
        out = rearrange(out, '(b nl) h ls d -> b h d (nl ls)', h=self.num_heads, b=B) # [B, h, d, S]
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        res = out.gather(dim=-1,index=q_idx_rev).reshape(B,D,S).permute(0,2,1) # [B, S, D]

        res = self.proj(res) # [B, S, D] add cpu
        res = self.proj_drop(res)

        res = x + res # [B, S, D]
        
        return res


    



'''
Date: 2022-03-12 11:47:58
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 14:05:49
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .PointConT_util import PatchAbstraction, ConT
from .ResMLP import MLPBlock1D, MLPBlockFC
from einops import rearrange


import sys
import numpy as np



class Backbone(nn.Module):
    def __init__(self,patch_dim,num_points,down_ratio,patch_size,local_size,num_heads ):
        super().__init__()
        self.nblocks = len(patch_dim) - 1
        self.patch_abstraction = nn.ModuleList()
        self.patch_transformer = nn.ModuleList()
        self.patch_embedding = nn.ModuleList()
        for i in range(self.nblocks):
            self.patch_abstraction.append(PatchAbstraction(int(num_points/down_ratio[i]), 
                                                           patch_size[i], 
                                                           2*patch_dim[i], 
                                                           [patch_dim[i+1], patch_dim[i+1]]))
            self.patch_transformer.append(ConT(patch_dim[i+1], local_size[i], num_heads))
            self.patch_embedding.append(MLPBlock1D(patch_dim[i+1]*2, patch_dim[i+1]))

    def forward(self, x):
        if x.shape[-1] == 5:
            pos = x
        else:
            pos = x[:, :, :5].contiguous()
        features = x
        pos_and_feats = []
        pos_and_feats.append([pos, features])

        for i in range(self.nblocks):
            #pos, max_features, avg_features, global_features = self.patch_abstraction[i](pos, features)
            pos, max_features, avg_features = self.patch_abstraction[i](pos, features)
            avg_features = self.patch_transformer[i](avg_features)
            #features = torch.cat([max_features, avg_features,global_features], dim=-1)
            features = torch.cat([max_features, avg_features], dim=-1).to("cuda:0")
            features = self.patch_embedding[i](features.transpose(1, 2)).transpose(1, 2).to("cuda:0")
            pos_and_feats.append([pos, features])

        return features, pos_and_feats



class PointConT_cls(nn.Module):
    def __init__(self, patch_dim=[5, 64, 128, 256, 512, 1024],num_points=2048,down_ratio=[2, 4, 8, 16, 32],patch_size=[16, 16, 16, 16, 16],local_size=[16, 16, 16, 16, 16] ,num_heads=4,dropout= 0.5,num_classes=60):
        super().__init__()
        self.backbone = Backbone(patch_dim,num_points,down_ratio,patch_size,local_size,num_heads)
        self.mlp1 = MLPBlockFC(patch_dim[-1], 512, dropout)
        self.mlp2 = MLPBlockFC(512, 256, dropout)
        self.output_layer = nn.Linear(256, num_classes)
        
    
    def forward_embeddings(self, x):
        b, c, img_v, img_t,m = x.size()
        #for i in range(m):
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_v = torch.arange(0, img_v, step=1).to("cuda:0") / (img_v - 1.0)#坐标归一化
        range_t = torch.arange(0, img_t, step=1).to("cuda:0") / (img_t - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_v, range_t, indexing='ij'), dim=-1).float().to("cuda:0")#len=64
        fea_pos = fea_pos
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(b*m, -1, -1, -1).to("cuda:0")#[8, 2, 64, 25]
        
        x = x.permute(0,4,1,2,3).reshape(b*m, c, img_v, img_t).to("cuda:0")
        x = torch.cat([x[torch.arange(b*m),:,:,:], pos], dim=1).to("cuda:0")#for循环优化
        x = x.reshape(b, m, c+2, img_v, img_t).permute(0,2,3,4,1)
        return x   
    
    def forward(self, x):
        n,c,T,V,M=x.size()
        x = self.forward_embeddings(x)#c,v,t
        x = x.reshape(n,c+2,-1).permute(0,2,1)
        patches, _ = self.backbone(x)  # [B, num_patches[-1], patch_dim[-1]]
        res = torch.max(patches, dim=1)[0].to("cuda:0")  # [B, patch_dim[-1]]
        res = self.mlp2(self.mlp1(res))
        res = self.output_layer(res) 

        return res
def PointConT(num_classes=60, **kwargs) -> PointConT_cls:#这里也改了！
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= PointConT_cls(patch_dim=[5, 64, 128, 256, 512, 1024],num_points=2048,down_ratio=[2, 4, 8, 16, 32],
                 patch_size=[16, 16, 16, 16, 16],local_size=[16, 16, 16, 16, 16] ,num_heads=4,
                 dropout= 0.5, num_classes=num_classes , **kwargs)
    model.to(device)
    from thop import profile


    inputs = torch.randn(1, 3, 64, 25,2).to(device)#.cuda()
    flops, params = profile(model, (inputs,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    return model

# def PointConT(num_classes=60, **kwargs) -> PointConT_cls:
#     return PointConT_cls(patch_dim=[5, 64, 128, 256, 512, 1024],num_points=2048,down_ratio=[2, 4, 8, 16, 32],
#                  patch_size=[16, 16, 16, 16, 16],local_size=[16, 16, 16, 16, 16] ,num_heads=4,
#                  dropout= 0.5, num_classes=num_classes , **kwargs)
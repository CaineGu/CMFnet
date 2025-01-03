import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
from decoder.dec_net import Decoder_Network
from encoder_dgcnn.dgcnn import DGCNN
from encoder_image.resnet import ResNet
from config import params
p = params()


class Featrue_Fusion(nn.Module):
    def __init__(self) -> None:
        super(Featrue_Fusion, self).__init__()
        self.img_encoder = ResNet()   
        self.convlayer1 = Convlayer1() 
        self.convlayer2 = Convlayer2()  
        self.convlayer3 = Convlayer3() 
        self.cross_attn1 = Cross_Attenton()
        self.cross_attn2 = Cross_Attenton()
        self.cross_attn3 = Cross_Attenton()
        
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        
    def forward(self, x_part, view):     
        im_feature   = self.img_encoder(view).permute(0, 2, 1)                      ## [16 * 256 * 196] ---> [16 * 196 * 256]
        pc_feature_1 = self.convlayer1(x_part[0]).permute(0, 2, 1)                  ## [16 * 256 * 128] ---> [16 * 128 * 256]
        pc_feature_2 = self.convlayer1(x_part[1]).permute(0, 2, 1)                  ## [16 * 256 * 128] ---> [16 * 128 * 256]
        pc_feature_3 = self.convlayer1(x_part[2]).permute(0, 2, 1)                  ## [16 * 256 * 128] ---> [16 * 128 * 256]
        
        feature1 = self.cross_attn1(pc_feature_1, im_feature)                       ## [16 * 128 * 256]
        feature2 = self.cross_attn2(pc_feature_2, im_feature)                       ## [16 * 128 * 256]
        feature3 = self.cross_attn3(pc_feature_3, im_feature)                       ## [16 * 128 * 256]
        
        feature1 = torch.unsqueeze(feature1, 1)                                     ## [16 * 1 * 128 * 256]
        feature2 = torch.unsqueeze(feature2, 1)                                     ## [16 * 1 * 128 * 256]
        feature3 = torch.unsqueeze(feature3, 1)                                     ## [16 * 1 * 128 * 256]
        
        latentfeature = torch.cat((feature1, feature2, feature3), 1)                ## [32 * 3 * 1280] 并联 转置
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))                 ## 用卷积代替了 MLP [16 * 1 * 128 * 256]
        latentfeature = torch.squeeze(latentfeature, 1)                             ## [16 * 128 * 256]
        
        return latentfeature                                      
       
                
class Cross_Attenton(nn.Module):
    def __init__(self) -> None:
        super(Cross_Attenton, self).__init__()
            
        self.cross_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(p.d_attn)

        self.self_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(p.d_attn)

        self.cross_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(p.d_attn)

        self.self_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(p.d_attn)

        self.cross_attn3 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm5 = nn.LayerNorm(p.d_attn)
    
    def forward(self, pc_feat, im_feat):
        
        x, _ = self.cross_attn1(pc_feat, im_feat, im_feat)            # [16 * 128 * 256]
        pc_feat = self.layer_norm1(x + pc_feat) 
        
        x, _ = self.self_attn1(pc_feat, pc_feat, pc_feat)
        pc_feat = self.layer_norm2(x + pc_feat)
        pc_skip = pc_feat
        
        x, _ = self.cross_attn2(pc_feat, im_feat, im_feat)
        pc_feat = self.layer_norm3(x + pc_feat)

        x, _ = self.self_attn2(pc_feat, pc_feat, pc_feat)
        pc_feat = self.layer_norm4(x + pc_feat)

        x, _ = self.cross_attn3(pc_feat, pc_skip, pc_skip)
        pc_feat = self.layer_norm5(x + pc_feat)                       # [16 * 128 * 256]
        
        return pc_feat


class Convlayer1(nn.Module):                ###  第 1 个分支的操作
    def __init__(self):
        super(Convlayer1, self).__init__()
        self.pc_encoder = DGCNN()       # 需要 [64 * 3 * 2048] 维度
        
    def forward(self, x_part):       
        pc_feat = self.pc_encoder(x_part)  
        
        return pc_feat


class Convlayer2(nn.Module):                ###  第 2 个分支的操作
    def __init__(self):
        super(Convlayer2, self).__init__()
        self.pc_encoder = DGCNN()
        
    def forward(self, x_part):
        pc_feat = self.pc_encoder(x_part)  
        
        return pc_feat


class Convlayer3(nn.Module):                ###  第 3 个分支的操作
    def __init__(self):
        super(Convlayer3, self).__init__()
        self.pc_encoder = DGCNN()
        
    def forward(self, x_part):
        pc_feat = self.pc_encoder(x_part) 
        
        return pc_feat


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.feature_fusion = Featrue_Fusion()
        self.pc_encoder = DGCNN()
        self.decoder = Decoder_Network()

    def forward(self, x_part, view):
        feature = self.feature_fusion(x_part, view)                ###  [16 * 128 * 256]
        x_part = x_part[0].permute(0, 2, 1)                        ###  [16 * 3 * 2048] ----> [16 * 2048 x 3]     

        final = self.decoder(feature, x_part)
            
        return final
        # return  feature

if __name__ == '__main__':
 
    x_part1 = torch.randn(16, 3, 2048).cuda()
    x_part2 = torch.randn(16, 3, 1024).cuda()
    x_part3 = torch.randn(16, 3, 512).cuda()
    x_part  = [x_part1, x_part2, x_part3]
    
    view = torch.randn(16, 3, 224, 224).cuda()
    model = Network().cuda()
    out = model(x_part, view)
    print(out.shape) 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")
    
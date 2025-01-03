from decoder.utils.utils import *
import numpy as np
import torch
import time

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src * dst^T = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn; [B, N, 1]
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm; [B, M, 1]
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src*dst^T

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))       #permute为转置,[B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)             #[B, N, M] + [B, N, 1]，dist的每一列都加上后面的列值
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)             #[B, N, M] + [B, 1, M],dist的每一行都加上后面的行值
    return dist

def Loss_function(complete_points, gt_points):
    
    gt_dist  = []
    cp_dist  = []
    distence = []
    base_points = None
    B, N, _  = gt_points.size()      
  
    idx_max= torch.argmax(gt_points, dim=1, keepdim=True)  # 获取最大值索引
    idx_min= torch.argmin(gt_points, dim=1, keepdim=True)
    
    for i in range(B):
        base_point = []
        gt_p = gt_points[i][:][:]
       
        pc_max = gt_p[idx_max[i]]        # 获得3个最大值
        pc_min = gt_p[idx_min[i]]        # 获得3个最小值
        base_point = torch.cat((pc_max, pc_min),dim=0).reshape(1,6,3)  # 将6个点拼接在一起
        
        # 拼成 （B,6,3)的形状
        if  base_points==None:
            base_points = base_point            
        else:
            base_points = torch.cat([base_points,base_point],dim=0)
        
           
    # 计算真实点云和补全点云到基准点的距离，得到（B,N,6）矩阵
    gt_dist = square_distance(gt_points, base_points)
    cp_dist = square_distance(complete_points, base_points)
    
    # 计算6点距离差总和，加到第7列
    gtdist_total = torch.sum(gt_dist, dim=2).reshape(B, N,1)  # 没问题
    cpdist_total = torch.sum(cp_dist, dim=2).reshape(B, N,1)  # 没问题
    
    # 将距离数组拼接为一个数组
    gt_dist = torch.cat((gt_dist, gtdist_total), dim=2)       # 没问题
    cp_dist = torch.cat((cp_dist, cpdist_total), dim=2)       # 没问题
    
    # 距离按照第7列排序（降序） 升序降序无所谓
    _, indices_gt = torch.sort(gt_dist, 1, descending=True)  # 排序操作别理解错了，
    _, indices_cp = torch.sort(cp_dist, 1, descending=True)
    indices_gt = indices_gt[:,:,-1].reshape(B,N)             # 索引变成二维，便于排序
    indices_cp = indices_cp[:,:,-1].reshape(B,N)
    gt_sorted = None
    cp_sorted = None
    
    # 得到降序的 batch_size
    for j in range(B):
        
        gt_dist_j = torch.index_select(gt_dist[j][:][:],0,indices_gt[j]).reshape(1,N,7)
        cp_dist_j = torch.index_select(cp_dist[j][:][:],0,indices_cp[j]).reshape(1,N,7)

        if gt_sorted == None:
            gt_sorted = gt_dist_j
            cp_sorted = cp_dist_j
        else:
            gt_sorted = torch.cat([gt_sorted,gt_dist_j],dim=0)
            cp_sorted = torch.cat([cp_sorted,cp_dist_j],dim=0) 
    
    # 对应作差
    distence = cp_sorted - gt_sorted
    
    # 矩阵每个元素求绝对值
    distence = distence.abs()
    
    # 计算总损失
    loss_dist = torch.sum(distence[:,:, 0 : 6])

    return loss_dist*1e-5

class L1_ChamferLoss(nn.Module):
    def __init__(self):
        super(L1_ChamferLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        # print(dist1, dist1.shape) [B, N]
        dist = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
        return dist / 2


if __name__ == '__main__':
    
    batch_size = 128
    points_num = 2048
    # #print(torch.rand(4, 5)) # 随机生成4*5的矩阵，值在0-1的浮点数
    # #print(torch.randint(2, 4, (3, 3)))  # (3,3)表示矩阵大小，2,4生成2-4（不包括4）的整数
    input_points = torch.randn(batch_size, points_num, 3).cuda()  # 将生成的np矩阵转换为tensor类型
    gt_points    = torch.randn(batch_size, points_num, 3).cuda()
   
    loss_distence = Loss_function(input_points,gt_points)
    print(f'距离是： {loss_distence}')
    
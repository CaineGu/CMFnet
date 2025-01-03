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
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) #permute为转置,[B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1) #[B, N, M] + [B, N, 1]，dist的每一列都加上后面的列值
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) #[B, N, M] + [B, 1, M],dist的每一行都加上后面的行值
    return dist

def Loss_function(complete_points, gt_points, base_num:int, alpha:float, belta:float):

    base_points = farthest_point_sample(gt_points, base_num).cuda()  # 最远点采样7个点
    gt_dist  = []
    cp_dist  = []
    distence = []
    B, N, _  = gt_points.size() # size()函数得到gt_points的矩阵形状
    
    # 计算真实点云和补全点云到基准点的距离，得到（1,2048,7）矩阵
    gt_dist = square_distance(gt_points, base_points).cuda()
    cp_dist = square_distance(complete_points, base_points).cuda()
    
    # 计算7点距离差总和，加到第8列
    gtdist_total = torch.sum(gt_dist, dim=2).reshape(N, 1).cuda()
    cpdist_total = torch.sum(cp_dist, dim=2).reshape(N, 1).cuda()
    
    # squeeze(0)降低维度，将（1,2048,7） 降为（2048,7）
    gt_dist = torch.cat((gt_dist.squeeze(0), gtdist_total ), dim=1).cuda()
    cp_dist = torch.cat((cp_dist.squeeze(0), cpdist_total), dim=1).cuda()
    
    # 按照第8列排序（升序） 升序降序无所谓
    gt_dist = gt_dist[gt_dist[:, base_num].argsort()].cuda()                     
    cp_dist = cp_dist[cp_dist[:, base_num].argsort()].cuda()
    
    # 对应作差
    distence = cp_dist - gt_dist
    
    # 矩阵每个元素求绝对值
    distence = distence.abs().cuda()
    
    # 计算总损失
    loss_dist = torch.sum(alpha * distence[:, 0 : base_num-1]) + torch.sum(belta * distence[ : , -1]).cuda()

    return loss_dist.item()

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
     
    t2 = time.time()
    for i in range(10000):   
        batch_size = 1
        points_num = 2048
        #print(np.random.rand(4, 5)) # 随机生成4*5的矩阵，值在0-1的浮点数
        #print(np.random.randint(2, 4, (3, 3)))  # (3,3)表示矩阵大小，2,4生成2-4（不包括4）的整数
        input_points = torch.tensor(np.array(np.random.randn(batch_size, points_num, 3),dtype=np.float32)).cuda()  # 将生成的np矩阵转换为tensor类型
        gt_points    = torch.tensor(np.array(np.random.randn(batch_size, points_num, 3),dtype=np.float32)).cuda()
    
    
        # reshape 增加维度 （B,N,3)
        #input_points = input_points.reshape([batch_size,2048,3])
        #gt_points = gt_points.reshape([1,2048,3])
        base_num     = 7
        alpha        = 0.5      # 各点距离和的系数
        belta        = 0.5      # 总距离和的系数
        loss_distence = Loss_function(input_points,gt_points, base_num,alpha,belta)
        #print(loss_distence)
    
    
    t3 = time.time()
    train_time2 = t3 -t2
    print(f'开始：{t2}, 结束：{t3}')
    print(f'我们方法计算10000次用时：{train_time2}')



    t0 = time.time()
    loss_cd = L1_ChamferLoss()  
    for j in range(10000):
        batch_size = 1
        points_num = 2048
        #print(np.random.rand(4, 5)) # 随机生成4*5的矩阵，值在0-1的浮点数
        #print(np.random.randint(2, 4, (3, 3)))  # (3,3)表示矩阵大小，2,4生成2-4（不包括4）的整数
        input_points = torch.tensor(np.array(np.random.randn(batch_size, points_num, 3),dtype=np.float32)).cuda()  # 将生成的np矩阵转换为tensor类型
        gt_points    = torch.tensor(np.array(np.random.randn(batch_size, points_num, 3),dtype=np.float32)).cuda()
        
        
        # reshape 增加维度 （B,N,3)
        #input_points = input_points.reshape([batch_size,2048,3])
        #gt_points = gt_points.reshape([1,2048,3])
        base_num     = 7
        alpha        = 0.5      # 各点距离和的系数
        belta        = 0.5      # 总距离和的系数
        loss_distence = loss_cd.forward(input_points, gt_points)
   
    t1 = time.time()
    train_time1 = t1 - t0
    print(f'开始：{t0}, 结束：{t1}')
    print(f"CD方法 10000次用时：{train_time1}。")
    


    
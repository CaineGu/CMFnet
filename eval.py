import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataloader import ViPCDataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
from model import Network
from decoder.utils.utils import *
from datetime import datetime, timedelta
import time 
# import open3d as o3d


opt = params()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ViPCDataset_test = ViPCDataLoader('/root/XMFnet-2080ti/dataset/test_list.txt', data_path=opt.dataroot, status= "test", category = "lamp")
test_loader = DataLoader(ViPCDataset_test,
                            batch_size=1,
                            num_workers=opt.nThreads,
                            shuffle=True,
                            drop_last=True)



model = Network().to(device)
model.load_state_dict(torch.load("/root/XMFnet-2080ti/best_lamp.pt")['model_state_dict'])
loss_eval = L2_ChamferEval_1000()
loss_f1 = F1Score()

with torch.no_grad():
    model.eval()
    i = 0
    Loss = 0 
    f1_final = 0
    for data in tqdm(test_loader):

        i += 1
        image = data[0].to(device)
        partial = data[2].to(device)
        gt = data[1].to(device)  

        partial = farthest_point_sample(partial, 2048)
        pt_pointcloud = partial.cpu()
        pt_pointcloud = pt_pointcloud.detach().numpy()
        pt_pointcloud = pt_pointcloud[0][:][:]      
        np.savetxt('/root/XMFnet-2080ti/pt.txt',pt_pointcloud)    
                          
        gt = farthest_point_sample(gt, 2048)
        gt_pointcloud = gt.cpu()
        gt_pointcloud = gt_pointcloud.detach().numpy()
        gt_pointcloud = gt_pointcloud[0][:][:]      
        np.savetxt('/root/XMFnet-2080ti/gt.txt',gt_pointcloud)    
            
        partial = partial.permute(0, 2, 1)

        complete = model(partial, image)
        cp_pointcloud = complete.cpu()
        cp_pointcloud = cp_pointcloud.detach().numpy()
        cp_pointcloud = cp_pointcloud[0][:][:]      
        np.savetxt('/root/XMFnet-2080ti/cp.txt',cp_pointcloud)    
         
        #Compute the eval loss
        loss = loss_eval(complete, gt)
        f1, _, _  = loss_f1(complete, gt)
        f1 = f1.mean()

        Loss += loss
        f1_final += f1
        

    Loss = Loss/i
    f1_final = f1_final/i



print(f"The evaluation loss for {opt.cat} is :{Loss}")
print(f"The F1-score for {opt.cat} is :{f1_final}")



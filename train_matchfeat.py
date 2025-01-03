import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
from decoder.utils.utils import *
from model import Network, GT_Network
from config import params
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import ViPCDataLoader
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import DataParallel
from all_utils import IOStream

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


opt = params()

if opt.cat != None:

    CLASS = opt.cat
else:
    CLASS = 'plane'


MODEL = 'model_supervised'
FLAG = 'train'
DEVICE = 'cuda:0'
VERSION = '4.16'
BATCH_SIZE = int(opt.batch_size)
MAX_EPOCH = int(opt.n_epochs)
EVAL_EPOCH = int(opt.eval_epoch)
RESUME = False


TIME_FLAG = time.asctime(time.localtime(time.time()))
CKPT_RECORD_FOLDER = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/record'
CKPT_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt.pth'
CONFIG_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/CONFIG.txt'



def save_ckpt(epoch, net, optimizer_all):
    ckpt = dict(
        epoch=epoch,
        model=net.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
    )
    torch.save(ckpt, CKPT_FILE)


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_one_step(data, optimizer, network):

    gt = data[1].to(device)
    image = data[0].to(device)
    
    gt = farthest_point_sample(gt, 2048)

    gt_permute = gt.permute(0, 2, 1)
    network.train()
    complete = network(gt_permute, image)
   
    loss_total = loss_cd(complete, gt)  ### 注意 gt 别被占用
    
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return loss_total


best_loss = 99999
best_epoch = 0
resume_epoch = 0
board_writer = SummaryWriter(
    comment=f'{MODEL}_{VERSION}_{BATCH_SIZE}_{FLAG}_{CLASS}_{TIME_FLAG}')

model = GT_Network().apply(weights_init_normal)
loss_cd = L1_ChamferLoss()
loss_cd_eval = L2_ChamferEval()
optimizer = torch.optim.Adam(filter(
    lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999))

ViPCDataset_train = ViPCDataLoader(
    '/root/XMFnet-2080ti/dataset/train_list2.txt', data_path=opt.dataroot, status="train", category=opt.cat)
train_loader = DataLoader(ViPCDataset_train,
                          batch_size=opt.batch_size,
                          num_workers=opt.nThreads,
                          shuffle=True,
                          drop_last=True)

ViPCDataset_test = ViPCDataLoader(
    '/root/XMFnet-2080ti/dataset/test_list2.txt', data_path=opt.dataroot, status="test", category=opt.cat)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=opt.batch_size,
                         num_workers=opt.nThreads,
                         shuffle=True,
                         drop_last=True)


if RESUME:
    ckpt_path = "/root/XMFnet-2080ti/ckpt_chair_146.pt"
    ckpt_dict = torch.load(ckpt_path)
    model.load_state_dict(ckpt_dict['model_state_dict'], strict=False)
    optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    resume_epoch = ckpt_dict['epoch']
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


if not os.path.exists(os.path.join(CKPT_RECORD_FOLDER)):
    os.makedirs(os.path.join(CKPT_RECORD_FOLDER))

with open(CONFIG_FILE, 'w') as f:
    f.write('RESUME:'+str(RESUME)+'\n')
    f.write('FLAG:'+str(FLAG)+'\n')
    f.write('DEVICE:'+str(DEVICE)+'\n')
    f.write('BATCH_SIZE:'+str(BATCH_SIZE)+'\n')
    f.write('MAX_EPOCH:'+str(MAX_EPOCH)+'\n')
    f.write('CLASS:'+str(CLASS)+'\n')
    f.write('VERSION:'+str(VERSION)+'\n')
    f.write(str(opt.__dict__))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'GPU的数量是：{torch.cuda.device_count()}')
gpus = [0,1]

model.train()
model.to(device)
model = DataParallel(model, device_ids=gpus, output_device=gpus[0])
# model = DDP(model, device_ids=[local_rank], output_device=local_rank)

print('--------------------')
print('Training Starting')
print(f'Training Class: {CLASS}')
print('--------------------')

set_seed()


io = IOStream('/root/XMFnet-2080ti/checkpoints/run.log')
io.cprint(str(opt))


for epoch in range(resume_epoch, resume_epoch + opt.n_epochs):

####################  训练  ###################

    if epoch < 2:
        opt.lr = 0.0001
    elif epoch < 20:
        opt.lr = 0.00001
    else:
        opt.lr = 0.000001
        

    Loss = 0
    i = 0
    
    loop = tqdm(train_loader,
                desc='train')  
    for data in loop:

        loss = train_one_step(data, optimizer, network=model)
        i += 1

        Loss += loss.item()
        loop.set_description(f'Epoch [{epoch+1}/{opt.n_epochs}]')
        loop.set_postfix({'loss': '{0:1.8f}'.format(Loss / i)})
    

    outstr = 'Train %d, loss: %.8f' % (epoch+1, Loss/i)
    io.cprint(outstr)
    
    
    with torch.no_grad():
       
        model.eval()
        i = 0
        Loss = 0
        Losscd=0
        
        loop2 = tqdm(test_loader,
                desc='test',
                colour="blue")
        for data in loop2:

            i += 1
            gt = data[1].to(device)
            image = data[0].to(device)
            
            gt = farthest_point_sample(gt, 2048)
            gt_permute = gt.permute(0, 2, 1)

            complete = model(gt_permute, image)

            loss = loss_cd_eval(complete, gt)
            losscd = loss_cd(complete, gt)   # 与训练CD一致 对比用
            
            Loss += loss.item()
            Losscd += losscd.item()
            
            # 更新训练信息
            loop2.set_description(f'Epoch [{epoch+1}/{opt.n_epochs}]')
            loop2.set_postfix({'losscd': '{0:1.8f}'.format(Losscd * 1.0 / i)},{'loss_eval': '{0:1.8f}'.format(Loss * 1.0 / i)})
        
        Loss_eval = Loss/i
        LOSSCD = Losscd/i
        
        # board_writer.add_scalar(
        #     "Average_Loss_epochs_test", Loss, epoch)

        if Loss_eval < best_loss:
            best_loss  = Loss_eval
            best_epoch = epoch+1
            LossCD     = LOSSCD
            torch.save(model.state_dict(), '/root/XMFnet-2080ti/checkpoints/ckpt_table_gt2.t7')
            # torch.save(model, f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt_cabinet_{epoch}.pt')
            print('model saved!')
    print('****************************')
    print("最佳训练轮次： ",best_epoch, ' ', '最佳损失为： ', best_loss, f'losscd 损失为： {LossCD}')
    print('****************************')
 
    outstr = 'best_epoch %d, best_loss: %.8f' % (best_epoch, best_loss)
    io.cprint(outstr)

print('Train Finished!!')

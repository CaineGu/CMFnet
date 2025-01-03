import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
from decoder.utils.utils import *



import scipy.io as sio

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def rotation_z(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                [sin_theta, cos_theta, 0.0],
                                [0.0, 0.0, 1.0]])
    return pts @ rotation_matrix.T


def rotation_y(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                [0.0, 1.0, 0.0],
                                [sin_theta, 0.0, cos_theta]])
    return pts @ rotation_matrix.T


def rotation_x(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos_theta, -sin_theta],
                                [0.0, sin_theta, cos_theta]])
    return pts @ rotation_matrix.T

class ViPCDataLoader(Dataset):
    def __init__(self,filepath,data_path,status,pc_input_num=3500, view_align=False, category='all'):
        super(ViPCDataLoader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align   # 图像视觉对准
        self.cat_map = {
            'plane':'02691156',    # 飞机
            'bench': '02828884',   # 长凳
            'cabinet':'02933112',  # 柜子
            'car':'02958343',      # 汽车
            'chair':'03001627',    # 椅子
            'monitor': '03211117', # 显示器
            'lamp':'03636649',     # 灯泡
            'speaker': '03691459', # 扬声器
            'firearm': '04090263', # 火器
            'couch':'04256520',    # 长沙发
            'table':'04379243',    # 桌子
            'cellphone': '04401088', # 手机
            'watercraft':'04530566' # 船只
        }
        with open(filepath, 'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')  # partial pc 路径
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')               # ground_true 路径
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')      # Image 路径

        for key in self.filelist:
            if category !='all':
                if key.split(';')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)

        self.transform = transforms.Compose([   # transforms.Compose 变换组合函数，对数据进行一系列变换
            transforms.Resize([224, 224]),     # 尺寸变换  将图片短边缩放至224，长宽比保持不变：本网络中要的是 224*224 故改为（[224，224]）
            transforms.ToTensor()       # 变换成tensor数据
        ])

        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):   # getitem 取得物品    通过索引返回图片。

        key = self.key[idx]
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split(';')[0]+'/'+ key.split(';')[1]+'/'+key.split(';')[-1].replace('\n', '')+'.dat')
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:  # view_align = False
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0, 23)).rjust(2, '0')  # rand.randint()方法返回指定范围内的整数
                    # rjust（2，‘0’）填充字符串，使生成的str文件名是2位，不足两位的补0.
                    #  key[:-3] 是对key字符串取值直到倒数第三个，到分号的位置。
       
        pc_path = os.path.join(self.gt_path, ran_key.split(';')[0]+'/'+ ran_key.split(';')[1]+'/'+ran_key.split(';')[-1].replace('\n', '')+'.dat')
        view_path = os.path.join(self.rendering_path,ran_key.split(';')[0]+'/'+ran_key.split(';')[1]+'/rendering/'+ran_key.split(';')[-1].replace('\n','')+'.png')
        # 以上两步是获取 图像和真实点云 相同的路径

        #Inserted to correct a bug in the splitting for some lines 
        if(len(ran_key.split(';')[-1])>3):
            #print("bug")  # 这里打印bug 是因为ran_key字符串最后（文件名）长度超过了3
            #print(ran_key.split(';')[-1])  # 打印文件出错的位置
            #print(f'找到出错位置了，在这：{ran_key}')
            fin = ran_key.split(';')[-1][-2:]
            interm = ran_key.split(';')[-1][:-2]
            
            pc_path = os.path.join(self.gt_path, ran_key.split(';')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            view_path = os.path.join(self.rendering_path,ran_key.split(';')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = self.transform(Image.open(view_path))
        views = views[:3,:,:]
        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # load gt
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]


        # load the view metadata
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()

    def __len__(self):
        return len(self.key)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    category = "table"
    ViPCDataset = ViPCDataLoader('test_list2.txt',data_path='/root/autodl-tmp/datapath',status='test', category = category)
    train_loader = DataLoader(ViPCDataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    
    


    for image, gt, partial in tqdm(train_loader):
        
        print(image.shape)
        
        pass
    
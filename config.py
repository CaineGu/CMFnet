



class params:
    def __init__(self): 
        
        #General Parameters
        self.cat = "couch"  # 类别是 全部类别

        
        #DGCNN parameters
        self.emb_dims = 256
        self.dropout = 0.5
        self.k = 20 
        
        #EdgePooling parameters
        self.k_pool1 = 16
        self.k_pool2 = 6
        self.pool1_points = 512
        self.pool2_points = 128
        self.scoring_fun = "tanh"

        #Decoder parameters 
        self.num_branch = 8
        self.K1 = 64
        self.K2 = 64
        self.N = 128
        self.method = "integrated"
        self.alpha = 0.01


        #Multi-head Attention parameters
        self.d_attn = 256
        self.num_heads = 4

        # Linear parameters
        self.in_num  = 1024
        self.out_num = 256

        #Training parameters
        self.batch_size = 128
        self.nThreads = 12  # num_workers = cup核数 性能拉满
        self.lr = 0.001
        self.dataroot = "/root/autodl-tmp/datapath"
        self.n_epochs = 260   
        self.ckp_dir = "./checkpoints_path"
        self.ckp_epoch = 5
        self.eval_epoch = 5
        self.resume = ''
        self.loss_print = 500
        self.vis_step = 1
        


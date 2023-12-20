from easydict import EasyDict as edict
import torch
cfg   = edict()


cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.img_size = 92 
cfg.mn,cfg.sd = ([0.5,0.5,0.5],[0.5,0.5,0.5])
cfg.batch_size = 32
cfg.epochs = 100
cfg.lr = 0.001
cfg.num_classes = 10
cfg.in_channels = 3
cfg.hidden_dim = 64
cfg.patch_size = 4
cfg.num_patches = (cfg.img_size // cfg.patch_size) ** 2
cfg.mlp_dim = 1024
cfg.num_heads = 8
cfg.num_layers = 4
cfg.dropout = 0.2

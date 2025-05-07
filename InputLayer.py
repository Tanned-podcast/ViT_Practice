#Cursorの使い方：Ctrl+Lでチャット，tabで補完，Ctrl+Kで生成
import torch
import torch.nn as nn

class VitInputLayer():
    def __init__ (self, num_patch_row: int = 2, in_channels: int = 3, embed_dim: int = 384, img_size: int = 32):
        super(VitInputLayer, self).__init__()
        self.num_patch_row = num_patch_row
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.img_size = img_size

        
        

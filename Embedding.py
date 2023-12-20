import torch
import torch.nn as nn
class projection(nn.Module):
  def __init__(self, in_channels, hidden_dim, patch_size):
    super().__init__()
    self.projection = nn.Conv2d(in_channels, hidden_dim, patch_size, patch_size)
  def forward(self, x):
    # input shape (batch_size, in_channels(3 or 1), img_size, img_size)
    # output shape (batch_size, num_patches, hidden_dim)
    x = self.projection(x)
    x = x.view(x.shape[0], -1, x.shape[1])
    
    return x

class Embedding(nn.Module):
  def __init__(self, num_patches, hidden_dim ):
    super().__init__()

    self.cls_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
  def forward(self, x):
    # input shape  : (batch_size, num_patches, hidden_dim)
    # output shape : (batch_size, num_patches + 1, hidden_dim)
    cls_embedding = self.cls_embedding.repeat(x.shape[0],1,1)
    x = torch.cat((cls_embedding, x), dim = 1)
    x = x + self.pos_embedding
    return x
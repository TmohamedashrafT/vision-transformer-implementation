import torch 
import torch.nn as nn
class attention_layer(nn.Module):
  def __init__(self,):
    super().__init__()
    self.act = nn.Softmax(dim = -1)
  def forward(self, q, k, v):
    # input shape  : (batch_size, head, num_patches + 1, head_size)
    # output shape : (batch_size, head, num_patches + 1, head_size)
    head_size = q.shape[-1]
    scale_matmul = torch.matmul(q, k.transpose(-1,-2)) / (head_size ** 0.5)
    prop = self.act(scale_matmul)
    out = torch.matmul(prop, v)
    return out

class Multi_Head_Attention(nn.Module):
  def __init__(self,hidden_dim, num_heads):
    super().__init__()
    self.q = nn.Linear(hidden_dim, hidden_dim)
    self.k = nn.Linear(hidden_dim, hidden_dim)
    self.v = nn.Linear(hidden_dim, hidden_dim)
    self.num_heads = num_heads
    self.attention_layer = attention_layer()
    self.linear = nn.Linear(hidden_dim, hidden_dim)
  def forward(self, x):
    # input shape  : (batch_size, num_patches + 1, hidden_dim)
    # output shape : (batch_size, num_patches + 1, hidden_dim)
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)
    ## using to_head and from_head to calculate heads in parallel
    q, k, v = self.to_head(q), self.to_head(k), self.to_head(v)
    attention_res = self.attention_layer(q, k, v)
    attention_res = self.from_head(attention_res)
    res = self.linear(attention_res)
    return res
  def to_head(self,x):
    ## output shape : (batch_size, num_heads, num_patches + 1, head_size)
    return x.view(x.shape[0], self.num_heads, x.shape[1], -1)
  def from_head(self,x):
    ## output shape : (batch_size, num_patches + 1, hidden_dim)
    return x.view(x.shape[0], -1, x.shape[1] * x.shape[-1])
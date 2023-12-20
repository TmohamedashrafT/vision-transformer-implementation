import torch.nn as nn
from Multi_Head_Attention import Multi_Head_Attention
class MLP(nn.Module):
  def __init__(self, hidden_dim, mlp_dim, dropout):
    super().__init__()
    self.linear_1 = nn.Linear(hidden_dim, mlp_dim)
    self.act = nn.GELU()
    self.dropout_1 = nn.Dropout(dropout)
    self.linear_2  = nn.Linear(mlp_dim, hidden_dim)
    self.dropout_2 = nn.Dropout(dropout)
  def forward(self, x):
    x = self.linear_1(x)
    x = self.act(x)
    x = self.dropout_1(x)
    x = self.linear_2(x)
    x = self.dropout_2(x)
    return x

class encoder_layer(nn.Module):
  def __init__(self,hidden_dim, mlp_dim, num_heads, dropout):
    super().__init__()
    self.norm_1 = nn.LayerNorm(hidden_dim)
    self.norm_2 = nn.LayerNorm(hidden_dim)
    self.mlp   = MLP(hidden_dim, mlp_dim, dropout)
    self.mha   = Multi_Head_Attention(hidden_dim, num_heads)
  def forward(self,x):
    input = x
    x = self.norm_1(x)
    x = self.mha(x)
    x = x + input
    input = x
    x = self.norm_2(x)
    x = self.mlp(x)
    x = x + input
    return x

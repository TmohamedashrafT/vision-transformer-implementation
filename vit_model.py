from Embedding import Embedding, projection
from Transformer_Encoder import encoder_layer
import torch.nn as nn
class vit(nn.Module):
  def __init__(self, num_classes,
               in_channels,
               hidden_dim,
               patch_size,
               num_patches,
               mlp_dim,
               num_heads,
               num_layers,
               dropout = 0.2):
    super().__init__()
    self.projection = projection(in_channels, hidden_dim, patch_size)
    self.Embedding  = Embedding(num_patches, hidden_dim)
    self.encoder_layers = nn.ModuleList([encoder_layer(hidden_dim, mlp_dim, num_heads, dropout) for _ in range(num_layers)])
    self.output = nn.Linear(hidden_dim, num_classes)
  def forward(self, x):
    x = self.projection(x)
    x = self.Embedding(x)
    for layer in self.encoder_layers:
      x = layer(x)
    x = self.output(x[:,0,:])
    return x




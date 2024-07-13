import torch.nn as nn
import torch


class CubeTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, num_moves):
        super(CubeTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(model_dim, num_moves)
    
    def forward(self, src, tgt):
        src = self.embedding(src)
        src = src.permute(1, 0, 2) 
        tgt = self.embedding(tgt)
        tgt = tgt.permute(1, 0, 2) 
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output.permute(1, 0, 2)

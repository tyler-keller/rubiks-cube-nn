import torch
import torch.nn as nn


class CubeTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, num_moves):
        super(CubeTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(model_dim, num_moves)
    
    def forward(self, src, tgt):
        src = self.embedding(src)
        src = src.permute(1, 0, 2) 
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output.permute(1, 0, 2)


class CubeNN(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_moves):
        super(CubeNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, model_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(model_dim, model_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(model_dim, num_moves))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

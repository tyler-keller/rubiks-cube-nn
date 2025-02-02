import torch
import torch.nn as nn


class CubeTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, model_dim, hidden_dim, num_layers, num_heads, num_moves, device='cpu'):
        super(CubeTransformer, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(model_dim, num_moves)
    
    def forward(self, src, tgt):
        src = self.embedding(src)
        src = src.permute(1, 0, 2) 
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output.permute(1, 0, 2)


class CubeRNN(nn.Module):
    def __init__(self, num_pieces, embedding_dim, hidden_size, output_size, num_layers=1, device='cpu'):
        super(CubeRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_pieces, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.embedding(x).to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        out = self.softmax(out)
        return out

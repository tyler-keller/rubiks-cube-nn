import torch
import torch.nn as nn


class CubeTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, hidden_dim, num_layers, num_heads, num_moves):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(model_dim, num_moves)
    
    def forward(self, src, tgt):
        src = self.embedding(src)
        src = src.permute(1, 0, 2) 
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output.permute(1, 0, 2)


class CubeRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x, lengths):
        out = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

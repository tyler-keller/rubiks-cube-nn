from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from pycuber.solver import CFOPSolver
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import pycuber as pc
from pycuber import *
import pandas as pd
import numpy as np
import random
import torch
import math
import os

from data import *
from model import *


def train_one_epoch(epoch_index, tb_writer, model, device, num_sequences):
    running_loss = 0.
    last_loss = 0.
    for i, sequence in enumerate(yield_sequences('../data/train_0.dat', num_sequences=num_sequences)):
        cube_states, move_states = sequence
        input_tensor = torch.stack(cube_states).to(device)
        output_tensor = torch.Tensor(move_states).to(torch.int64).to(device)
        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = loss_fn(outputs, output_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss += loss.item()
        if i % (num_sequences//5) == 0:
            last_loss = running_loss / num_sequences 
            print(f'    batch {i} loss: {last_loss}')
            tb_x = epoch_index * num_sequences + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss


move_mapping = {
    'U': 0, 'U\'': 1,
    'L': 2, 'L\'': 3,
    'F': 4, 'F\'': 5,
    'R': 6, 'R\'': 7,
    'B': 8, 'B\'': 9,
    'D': 10, 'D\'': 11,
    '$': 12
}

model_dim = 2056
output_dim = len(move_mapping)
num_layers = 4
num_pieces = 26 
embedding_dim = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = CubeTransformer(input_dim, model_dim, num_layers, num_heads, num_moves)
model = CubeRNN(num_pieces=num_pieces, embedding_dim=embedding_dim, hidden_size=model_dim, output_size=output_dim, num_layers=num_layers, device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if torch.cuda.is_available():
    model.cuda()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/rubiks_cube_nn_{}'.format(timestamp))

EPOCHS = 25 

best_vloss = 1_000_000.

print(f'Device: {device}')
print()

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch + 1}:')
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer, model, device, num_sequences=250)
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(yield_sequences('../data/train_12.dat', num_sequences=50)):
            cube_states, move_states = vdata
            vinput_tensor = torch.stack(cube_states).to(device)
            voutput_tensor = torch.Tensor(move_states).to(torch.int64).to(device)
            voutputs = model(vinput_tensor)
            vloss = loss_fn(voutputs, voutput_tensor)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = '../models/model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)
    epoch += 1
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

def encode_cube(cube):
    piece_to_index_mapping = {}
    location_to_array_position_mapping = {}
    cube = pc.Cube()
    index = 0
    for c in cube:
        piece_to_index_mapping[tuple([str(square[1]) for square in c[1]])] = index
        location_to_array_position_mapping[c[0]] = index
        index += 1
    cube_array = [None for _ in range(26)]
    for i, cubie_tuple in enumerate(cube):
        location, cubie = cubie_tuple
        squares = []
        for square in cubie:
            squares.append(str(square[1]))
        squares = tuple(squares)
        cube_array[location_to_array_position_mapping[location]] = piece_to_index_mapping[squares]
    return torch.Tensor([c for c in cube_array]).to(torch.int64)


def encode_move(move):
    move_mapping = {
        'U': 0, 'U\'': 1,
        'L': 2, 'L\'': 3,
        'F': 4, 'F\'': 5,
        'R': 6, 'R\'': 7,
        'B': 8, 'B\'': 9,
        'D': 10, 'D\'': 11,
        '$': 12
    }
    return move_mapping[move]


def random_line(filename, prev_lines):
    with open(filename, 'r') as file:
        rand_line = next(file)
        rand_num = 0
        for num, line in enumerate(file, 2):
            if random.randrange(num) or num in prev_lines:
                continue
            rand_line = line
            rand_num = num
        return rand_num, rand_line

def yield_sequences(filename, num_sequences=1000):
    i = 0
    prev_lines = []
    while i < num_sequences:
        line_num, line = random_line(filename, prev_lines)
        prev_lines.append(line_num)
        cubies_string, solution_string = line.split('|')
        mapping = {'U2': 'U U', 'L2': 'L L', 'R2': 'R R', 'F2': 'F F', 'D2': 'D D', 'B2': 'B B'}
        for k, v in mapping.items():
            solution_string = solution_string.replace(k, v)
        cubies_set = set([eval(cubie_string) for cubie_string in cubies_string.split(';')])
        cube = pc.Cube(cubies_set)
        moves = solution_string.split()
        cube_states = []
        move_states = []
        for move in moves:
            cube_states.append(encode_cube(cube))
            move_states.append(encode_move(move))
            cube.perform_step(move)
        cube_states.append(encode_cube(cube))
        move_states.append(encode_move('$'))
        if i % 100 == 0:
            print(f'Sequnce {i} of {num_sequences}')
        i += 1
        sequence = cube_states, move_states
        yield sequence


def train_one_epoch(epoch_index, tb_writer, model):
    running_loss = 0.
    last_loss = 0.
    num_sequences = 1000
    for i, sequence in enumerate(yield_sequences('../data/train_0.dat', num_sequences=num_sequences)):
        cube_states, move_states = sequence
        input_tensor = torch.stack(cube_states)
        output_tensor = torch.Tensor(move_states).to(torch.int64)
        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = loss_fn(outputs, output_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 
            print(f'    batch {i + 1} loss: {last_loss}')
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

model_dim = 512
output_dim = len(move_mapping)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = CubeTransformer(input_dim, model_dim, num_layers, num_heads, num_moves)
model = CubeRNN(num_pieces=26, embedding_dim=16, hidden_size=model_dim, output_size=output_dim, num_layers=1, device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch + 1}:')
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer, model)
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(yield_sequences('../data/train_12.dat', num_sequences=500)):
            cube_states, move_states = vdata
            vinput_tensor = torch.stack(cube_states)
            voutput_tensor = torch.Tensor(move_states).to(torch.int64)
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
        model_path = 'model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)
    epoch += 1
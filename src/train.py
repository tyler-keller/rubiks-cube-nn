from torch.utils.data import Dataset, DataLoader
from pycuber.solver import CFOPSolver
import torch.optim as optim
import torch.nn as nn
import pycuber as pc
from pycuber import *
import pandas as pd
import numpy as np
import torch
import math
import os

from data import CubeDataset
from model import CubeTransformer


move_mapping = {
    'U': 0, 'U\'': 1, 'U2': 2,
    'L': 3, 'L\'': 4, 'L2': 5,
    'F': 6, 'F\'': 1, 'F2': 8,
    'R': 9, 'R\'': 1, 'R2': 11,
    'B': 12, 'B\'': 1, 'B2': 14,
    'D': 15, 'D\'': 1, 'D2': 17, '$': 18
}


def convert_string_state_to_cube(string_state) -> Cube:
    '''
    Convert the string state to a Pycube object.
    ULFRBD ordered. 6 x 9 x 6 state encoding.
    '''
    cubie_set = set([eval(cubie_string) for cubie_string in string_state.split(';')])
    return pc.Cube(cubie_set)

def convert_cube_to_state(cube):
    '''
    Convert a cube object to the NN state representation.
    ULFRBD ordered. 6 x 9 x 6 state encoding.
    '''
    color_vectors = {
        'b': np.array([1, 0, 0, 0, 0, 0]),
        'g': np.array([0, 1, 0, 0, 0, 0]),
        'o': np.array([0, 0, 1, 0, 0, 0]),
        'r': np.array([0, 0, 0, 1, 0, 0]),
        'w': np.array([0, 0, 0, 0, 1, 0]),
        'y': np.array([0, 0, 0, 0, 0, 1]),
    }
    state = np.zeros(shape=(6, 9, 6))
    for i, face in enumerate('ULFRBD'):
        unpacked_face = [str(x)[1] for x in np.array(cube.get_face(face)).flatten()]
        for j, square in enumerate(unpacked_face):
            state[i, j] = color_vectors[square]
    return state.flatten()


def load_sequences_in_batches(filename, batch_size=1000):
    with open(filename, 'r') as f:
        batch = []
        for i, line in enumerate(f):
            string_state, solution = line.strip().split('|')
            unsolved_cube = convert_string_state_to_cube(string_state)
            sequence = []
            for step in solution.split():
                sequence.append((convert_cube_to_state(unsolved_cube), step))
                unsolved_cube.perform_step(step)
            sequence.append((convert_cube_to_state(unsolved_cube), '$'))
            batch.append(sequence)
            if (i + 1) % batch_size == 0:
                yield batch
                batch = []
        if batch:
            yield batch

input_dim = 6 * 9 * 6
model_dim = 512
num_layers = 6
num_heads = 8
num_moves = len(move_mapping)

model = CubeTransformer(input_dim, model_dim, num_layers, num_heads, num_moves)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
log_epoch = 10

for batch in load_sequences_in_batches('../data/train_0.dat', batch_size=1000):
    max_length = max(len(seq) for seq in batch)
    dataset = CubeDataset(batch, move_mapping, max_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    for batch in dataloader:
        print(batch)
        states, moves = batch
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, moves)
        loss.backward()
        optimizer.step()
        if num_epochs % log_epoch == 0:
            print(f'Epochs: {num_epochs} Loss: {loss}')
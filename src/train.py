from torch.utils.data import Dataset, DataLoader
from pycuber.solver import CFOPSolver
import torch.optim as optim
import torch.nn as nn
import pycuber as pc
import pandas as pd
import numpy as np
import torch
import math
import os

from data import CubeDataset
from model import CubeTransformer


# TODO: only if required... time-consuming to figure out reconstructing from Square objects...
def convert_string_state_to_cube(state):
    '''
    Convert the state representation to a Pycube object.
    ULFRBD ordered. 6 x 9 x 6 state encoding.
    '''
    color_vectors = {
        np.array([1, 0, 0, 0, 0, 0]): 'blue',
        np.array([0, 1, 0, 0, 0, 0]): 'green',
        np.array([0, 0, 1, 0, 0, 0]): 'orange',
        np.array([0, 0, 0, 1, 0, 0]): 'red',
        np.array([0, 0, 0, 0, 1, 0]): 'white',
        np.array([0, 0, 0, 0, 0, 1]): 'yellow',
    }
    cube = pc.Cube()
    for i, face in state:
        for j, vector in enumerate(face):
            square = color_vectors[vector]


move_mapping = {
    'U': 0, 'U\'': 1, 'U2': 2,
    'L': 3, 'L\'': 4, 'L2': 5,
    'F': 6, 'F\'': 1, 'F2': 8,
    'R': 9, 'R\'': 1, 'R2': 11,
    'B': 12, 'B\'': 1, 'B2': 14,
    'D': 15, 'D\'': 1, 'D2': 17, '$': 18
}


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



with open('../data/training.dat') as f:
    sequences = []
    for line in f:
        string_state, solution = line.split(',')
        cube = convert_string_state_to_cube(string_state)
        sequence = []
        unsolved_cube = ''
        for step in str(solution).split():
            sequence.append(unsolved_cube, step)
            f.write()
            unsolved_cube.perform_step(step)
        sequence.append((convert_cube_to_state(unsolved_cube), '$'))
        sequences.append(sequence)

max_length = max(len(seq) for seq in sequences)
dataset = CubeDataset(sequences, move_mapping, max_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
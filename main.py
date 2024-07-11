from torch.utils.data import Dataset, DataLoader
from pycuber.solver import CFOPSolver
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pycuber as pc
import pandas as pd
import numpy as np
import torch
import math
import os


# TODO: only if required... time-consuming to figure out reconstructing from Square objects...
# def convert_state_to_cube(state):
#     '''
#     Convert the state representation to a Pycube object.
#     ULFRBD ordered. 6 x 9 x 6 state encoding.
#     '''
#     color_vectors = {
#         np.array([1, 0, 0, 0, 0, 0]): 'blue',
#         np.array([0, 1, 0, 0, 0, 0]): 'green',
#         np.array([0, 0, 1, 0, 0, 0]): 'orange',
#         np.array([0, 0, 0, 1, 0, 0]): 'red',
#         np.array([0, 0, 0, 0, 1, 0]): 'white',
#         np.array([0, 0, 0, 0, 0, 1]): 'yellow',
#     }
#     cube = pc.Cube()
#     for i, face in state:
#         for j, vector in enumerate(face):
#             square = color_vectors[vector]


class CubeDataset(Dataset):
    def __init__(self, sequences, move_mapping, max_length):
        self.sequences = sequences
        self.move_mapping = move_mapping
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        states, moves = zip(*sequence)
        
        padded_states = np.zeros((self.max_length, len(states[0])))
        padded_moves = np.full(self.max_length, -1)  
        mask = np.zeros(self.max_length, dtype=bool)
        
        seq_length = len(states)
        padded_states[:seq_length] = states
        padded_moves[:seq_length] = [self.move_mapping[move] for move in moves]
        mask[:seq_length] = 1
        
        return padded_states, padded_moves, mask


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
    return state


class CubeTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, num_moves):
        super(CubeTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(model_dim, num_moves)
    
    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2) 
        output = self.transformer(src)
        output = self.fc_out(output)
        return output


def generate_sequences(n):
    sequences = []
    for _ in n:
        sequence = []
        cube = pc.Cube()
        alg = pc.Formula()
        random_alg = alg.random()
        cube(random_alg)
        unsolved_cube = cube.copy()
        solver = CFOPSolver(cube)
        solution = solver.solve(suppress_progress_messages=True)
        for step in str(solution).split():
            sequence.append((convert_cube_to_state(unsolved_cube), step))
            unsolved_cube.perform_step(step)
        sequences.append(sequence)
    return sequences


sequences = generate_sequences(1000)
move_mapping = {
    'U': 0, 'U\'': 1, 
    'L': 2, 'L\'': 3,
    'F': 4, 'F\'': 5,
    'R': 6, 'R\'': 7,
    'B': 8, 'B\'': 9,
    'D': 10, 'D\'': 11,
    '$': 12
} 
max_length = max(len(seq) for seq in sequences)
dataset = CubeDataset(sequences, move_mapping, max_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = 6 * 9 * 6
model_dim = 512
num_layers = 6
num_heads = 8
num_moves = 12

model = CubeTransformer(input_dim, model_dim, num_layers, num_heads, num_moves)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for batch in dataloader:
        states, moves = batch
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, moves)
        loss.backward()
        optimizer.step()
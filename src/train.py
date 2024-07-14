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

from data import *
from model import *


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


def load_sequences(filename, move_mapping, num_sequences=1000):
    '''Load a train.dat file and transform it into a series of cube states to move sequences.
    '''
    with open(filename, 'r') as f:
        print(f'OPENING FILE: {filename}')
        log_i = 100
        sequences = []
        for i, line in enumerate(f):
            if i == num_sequences:
                return sequences
            if i % log_i == 0:
                print(f'LINE: {i}')
            string_state, solution = line.strip().split('|')
            unsolved_cube = convert_string_state_to_cube(string_state)
            sequence = []
            for step in solution.split():
                sequence.append((convert_cube_to_state(unsolved_cube), move_mapping[step]))
                unsolved_cube.perform_step(step)
            sequence.append((convert_cube_to_state(unsolved_cube), move_mapping['$']))
            sequences.append(sequence)
        return sequences


def train(dataloader: DataLoader):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += (
            (pred >= 0.5).float() == label_batch
        ).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate(dataloader: DataLoader):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += (
                (pred >= 0.5).float() == label_batch
            ).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

move_mapping = {
    'U': 0, 'U\'': 1, 'U2': 2,
    'L': 3, 'L\'': 4, 'L2': 5,
    'F': 6, 'F\'': 1, 'F2': 8,
    'R': 9, 'R\'': 1, 'R2': 11,
    'B': 12, 'B\'': 1, 'B2': 14,
    'D': 15, 'D\'': 1, 'D2': 17, '$': 18
}

input_dim = 6 * 9 * 6
model_dim = 512
output_dim = len(move_mapping)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = CubeTransformer(input_dim, model_dim, num_layers, num_heads, num_moves)
model = CubeRNN(input_dim, model_dim, output_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
log_epoch = 10

sequences = load_sequences('../data/train_0.dat', move_mapping=move_mapping, num_sequences=100)

dataset = CubeDataset(sequences=sequences, move_mapping=move_mapping, max_length=max([len(x) for x in sequences]))
dataloader = DataLoader(dataset=dataset)

for sequences, moves in dataloader:
    print(sequences[0], moves[0])
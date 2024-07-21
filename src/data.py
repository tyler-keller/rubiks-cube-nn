from torch.utils.data import Dataset, DataLoader
from pycuber import *
import pycuber as pc
import numpy as np
import random
import torch

def encode_cube(cube):
    piece_to_index_mapping = {}
    location_to_array_position_mapping = {}
    mapping_cube = pc.Cube()
    index = 0
    for c in mapping_cube:
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
    return torch.Tensor([c / 25. for c in cube_array]).to(torch.int64)


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

def yield_cross_sequences(filename, num_sequences=1000):
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
        # if i % 100 == 0:
        #     print(f'Sequnce {i} of {num_sequences}')
        i += 1
        sequence = cube_states, move_states
        yield sequence

def yield_full_sequences(filename, num_sequences=1000):
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
        # if i % 100 == 0:
        #     print(f'Sequnce {i} of {num_sequences}')
        i += 1
        sequence = cube_states, move_states
        yield sequence

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
        padded_moves[:seq_length] = moves
        mask[:seq_length] = 1
        
        return padded_states, padded_moves

from pycuber.solver import CFOPSolver
import pycuber as pc
import pandas as pd
import numpy as np
import torch
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


class CubeDataset(torch.utils.data.Dataset):
    def __init__(self, state, next_move):
        self.state = state
        self.next_move = next_move

    def __getitem__(self, index) -> any:
        return self.state[index], self.next_move[index]


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


def generate_scramble_to_solve():
    cube = pc.Cube()
    alg = pc.Formula()
    random_alg = alg.random()
    cube(random_alg)
    original_cube = cube.copy()
    solver = CFOPSolver(cube)
    solution = solver.solve(suppress_progress_messages=True)
    for step in str(solution).split():
        original_cube.perform_step(step)
    print(cube)
    print(original_cube)


generate_scramble_to_solve()
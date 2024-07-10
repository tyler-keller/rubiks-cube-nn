from torch.utils.data import Dataset, DataLoader
import pycuber as pc
from pycuber.solver import CFOPSolver
import pandas as pd
import numpy as np
import torch
import os


class CubeDataset(Dataset):
    def __init__(self, state, next_move):
        self.state = state
        self.next_move = next_move

    def __getitem__(self, index) -> any:
        return self.state[index], self.next_move[index]


cube = pc.Cube()
alg = pc.Formula()

random_alg = alg.random()

cube(random_alg)
print(cube)

solver = CFOPSolver(cube)

solution = solver.solve(suppress_progress_messages=True)

print(cube)
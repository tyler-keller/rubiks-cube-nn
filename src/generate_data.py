import pycuber as pc
from pycuber.solver import CFOPSolver
import numpy as np
import os


def convert_cube_to_string(cube):
    squares = []
    for i, face in enumerate('ULFRBD'):
        unpacked_face = [str(x)[1] for x in np.array(cube.get_face(face)).flatten()]
        for j, square in enumerate(unpacked_face):
            squares.append(square)
    return ' '.join(squares)


os.makedirs('../data', exist_ok=True)
with open('../data/train.dat', 'w+') as f:
    sequence = ''
    log_i = 100
    n = 10_000
    for i in range(n):
        sequence = []
        cube = pc.Cube()
        alg = pc.Formula()
        random_alg = alg.random()
        cube(random_alg)
        unsolved_cube = cube.copy()
        solver = CFOPSolver(cube)
        solution = solver.solve(suppress_progress_messages=True)
        if i % log_i == 0:
            print(f'Sequence {i} of {n}...')

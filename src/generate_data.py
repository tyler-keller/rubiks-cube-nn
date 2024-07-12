import pycuber as pc
from pycuber import *
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


def convert_cube_to_cubie_strings(cube):
    return ';'.join([str(cubie[1]) for cubie in cube])


os.makedirs('../data', exist_ok=True)
for bruh in range(70):
    print(f'file {bruh} of 69')
    with open(f'../data/train_{bruh}.dat', 'w+') as f:
        log_i = 500
        n = 10_000
        for i in range(n):
            cube = pc.Cube()
            alg = pc.Formula()
            random_alg = alg.random()
            cube(random_alg)
            unsolved_cube = cube.copy()
            solver = CFOPSolver(cube)
            solution = solver.solve(suppress_progress_messages=True)
            f.write(f'{convert_cube_to_cubie_strings(unsolved_cube)}|{solution}\n')
            if i % log_i == 0:
                print(f'Sequence {i} of {n}...')

# rubiks-cube-nn

Side project that I want to put on my website.

Given a random scramble, solve the cube.

I'd love to show the cube in 3.js, the NN graph w/ corresponding weights lighting up, and lastly moves taken to solve the cube.

## Data//Model Representation

The Rubik's cube state is represented as a 6 x 9 x 6 tensor.

6 faces, 9 squares per face, size 6 state vector to represent color.

We'll start off w/ a standard dense NN.

The input dims will be the same as described above.

There will be hidden dims -- obviously.

The output dims will represent the next move the model should make.

There are 18 possible moves you can make in any state.

F, R, U, L, B, D, F', R', U', L', B', D', F2, R2, U2, L2, B2, D2

Lastly, we will use `$` to indicate a solved cube.

## Training

I'll be using the Pycuber library to generate training data.

Both for random scrambles and CFOP solutions.

#### Shoutouts to below:

https://github.com/kongaskristjan/rubik

https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions/data

https://github.com/adrianliaw/PyCuber

## Random Thoughts

This could lend itself well to the transformer architecture since it's kinda just sequence modeling.

## TODO

<!-- - convert pycube object to state representation -->

- generate dataset
    <!-- - state to next move -- then next state to next move -- 'til the cube is solved -->
    - set it up w/ the nice pytorch loader things

- train the model
    - i need something to figure out the positional relationship of moves... transformer time???
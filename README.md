# rubiks-cube-nn

Side project that I want to put on my website.

Given a random scramble, solve the cube.

## Data//Model Representation

The Rubik's cube state is represented as a 6 x 9 x 6 tensor.

6 faces, 9 squares per face, size 6 state vector to represent color.

The input dims will be the same as described above.

There will be hidden dims -- obviously.

The output dims will represent the next move the model should make.

There are 18 possible moves you can make in any state.

`F`, `R`, `U`, `L`, `B`, `D`, `F'`, `R'`, `U'`, `L'`, `B'`, `D'`, `F2`, `R2`, `U2`, `L2`, `B2`, `D2`

Lastly, we will use `$` to indicate a solved cube.


In the context, we're trying to train the model on an input sequence of cube states to an output move class.

W/ a block size of 8, starting from a completely solved cube -- we're taking the first input and saying if this is the first state you should make this move, then, on the second input, if this is the first state and then this is the second state you should make this move, so on and so on 'til the 8th move. Then we start back over.

I'm not sure what will end up working the best -- feeding in an entire cube state to move state sequence per batch (this will be the first I try) or some subset of moves in a scramble to solve state sequence.

The issue w/ an entire cube state to move state sequence is they're different lengths. I'd have to use some type of padding.


My goal is to train on enough data that we start to see CFOP-esque moves emerge from the weights.

## Training

I'll be using the Pycuber library to generate training data.

Both for random scrambles and CFOP solutions.

## Dataset

https://www.kaggle.com/datasets/tylerkeller/rubiks-cube-dataset/data

#### Shoutouts to below:

https://github.com/kongaskristjan/rubik

https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions/data

https://github.com/adrianliaw/PyCuber

## TODO

<!-- - convert pycube object to state representation -->

- generate dataset
    <!-- - state to next move -- then next state to next move -- 'til the cube is solved -->
    - set it up w/ the nice pytorch loader things

- train the model
    - i need something to figure out the positional relationship of moves... transformer time???
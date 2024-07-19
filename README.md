# rubiks-cube-nn

Side project that I want to put on my website.

Given a random scramble, solve the cube.

## Data//Model Representation

The Rubik's cube state is represented as a 1-D Tensor of cube pieces. The solved state cooresponds to the .

The input dims will be the same as described above.

There will be hidden dims -- obviously.

The output dims will represent the next move the model should make.

There are 18 possible moves you can make in any state.

`F`, `R`, `U`, `L`, `B`, `D`, `F'`, `R'`, `U'`, `L'`, `B'`, `D'`, `F2`, `R2`, `U2`, `L2`, `B2`, `D2`

However, we will reduce this down to 12 possible moves, where all double turns are changed to 2 single turns.

`F`, `R`, `U`, `L`, `B`, `D`, `F'`, `R'`, `U'`, `L'`, `B'`, `D'`

Lastly, we will use `$` to indicate a solved cube.


In context, we're trying to train the model on an input sequence of cube states to an output move classification.

W/ a block size of 8, starting from a completely solved cube -- we're taking the first input and saying if this is the first state you should make this move, then, on the second input, if this is the first state and then this is the second state you should make this move, so on and so on 'til the 8th move. Then we start back over.

I'm not sure what will end up working the best -- feeding in an entire cube state to move state sequence per batch (this will be the first I try) or some subset of moves in a scramble to solve state sequence.

Another option might be to break down the stages of cube solving -- Cross, F2L, OLL, PLL -- and train separate models to complete each of these tasks... could be interesting.

The issue w/ an entire cube state to move state sequence is they're different lengths. I'd have to use some type of padding.


My goal is to train on enough data that we start to see CFOP-esque moves emerge from the weights.


Our sequence data is many-to-one. Input is a sequence of cube states. Output is a class label (indicating what move to take).


## Training

I'll be using the Pycuber library to generate training data.

Both for random scrambles and CFOP solutions.

## Dataset

https://www.kaggle.com/datasets/tylerkeller/rubiks-cube-dataset/data

#### Shoutouts to below:

https://github.com/kongaskristjan/rubik

https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions/data

https://github.com/adrianliaw/PyCuber

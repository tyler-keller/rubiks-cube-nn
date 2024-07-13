from torch.utils.data import Dataset, DataLoader
import numpy as np

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

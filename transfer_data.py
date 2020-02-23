import torch
import os
import numpy as np

def process_inputs(X, Y):
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()
    X, Y = to_soduku(X, Y, is_input)
    return X, Y

def to_soduku(X, Y, is_input):
    soduku_X = X.argmax(dim=3) + 1
    soduku_Y = Y.argmax(dim=3) + 1
    is_input = is_input.permute(0,3,1,2)[:,0,:,:]
    soduku_X = soduku_X * is_input
    return soduku_X.numpy(), soduku_Y.numpy()

if __name__ == "__main__":
    data_dir = "sudoku"
    with open(os.path.join(data_dir, 'features.pt'), 'rb') as f:
        X_in = torch.load(f)
    with open(os.path.join(data_dir, 'labels.pt'), 'rb') as f:
        Y_in = torch.load(f)

    X_numpy, Y_numpy = process_inputs(X_in, Y_in)
    print(X_numpy[0])
    print(Y_numpy[0])
    np.save(os.path.join(data_dir, 'features.npy'), X_numpy)
    np.save(os.path.join(data_dir, 'labels.npy'), Y_numpy)

import torch
import os
import numpy as np
from src.solver import solver as SL

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

def np_back_torch(X):
    n = X.shape[0]
    print(n)
    assert X.shape[1] == 9 and X.shape[2] == 9
    X_tensor = torch.zeros(size=(n, 9, 9, 9), dtype=torch.float32)
    for t, one in enumerate(X):
        print(t)
        for i, row in enumerate(one):
            for j, num in enumerate(row):
                num = int(num)
                if num > 0:
                    X_tensor[t, i, j, num-1] = 1
    return X_tensor

if __name__ == "__main__":
    difficulty = ""
    back_dir = "sudoku" + difficulty
    # back_dir = "sudoku"

    with open(os.path.join(back_dir, 'features.pt'), 'rb') as f:
        X_in = torch.load(f)
    with open(os.path.join(back_dir, 'labels.pt'), 'rb') as f:
        Y_in = torch.load(f)
    # # #
    X_numpy, Y_numpy = process_inputs(X_in, Y_in)
    print(X_numpy[0])
    print(Y_numpy[0])

    np.save(os.path.join(back_dir, difficulty+'.npy'), X_numpy)
    np.save(os.path.join(back_dir, difficulty+'label.npy'), Y_numpy)
    exit(0)

    file_X = difficulty + ".npy"
    file_Y = difficulty + "_label.npy"
    X = np.load(os.path.join(back_dir, file_X))
    features = np_back_torch(X)
    torch.save(features, os.path.join(back_dir, "features.pt"))


    Y = np.zeros(shape=X.shape)
    for i,_ in enumerate(Y):
        print(i)
        Y[i] = SL.SudoSolveIt2(X[i], [], 1)

    labels = np_back_torch(Y)
    torch.save(labels, os.path.join(back_dir, "labels.pt"))

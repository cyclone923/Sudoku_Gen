from src.solver import solver as SL

import numpy as np
import torch
import os
from copy import deepcopy

data_set = "sudoku"
difficulty = ""
feature = np.load(os.path.join(data_set+difficulty, difficulty+".npy"))
label = np.load(os.path.join(data_set+difficulty, difficulty+"_label.npy"))

def to_action(fill):
    assert fill.shape == (9,9)
    assert np.count_nonzero(fill) == 1
    action = torch.zeros(size=(9,9,9))
    n_row, n_col = np.where(fill > 0)
    num = fill[n_row, n_col]
    action[num-1, n_row, n_col] = 1
    return action.flatten()

def to_state(m):
    assert m.shape == (9,9)
    state = torch.zeros(size=(10,9,9))
    for n_row, row in enumerate(m):
        for n_col, num in enumerate(row):
            state[num, n_row, n_col] = 1
    return state.flatten()


def solvabel(m):
    A = deepcopy(m)
    return np.min(SL.SudoSolveIt2(A, [], 1)) > 0

s_t = []
s_t_plus_one = []
a_t = []
scope = []

for i, f in enumerate(feature):
    print(i)
    assert solvabel(f)
    while np.min(f) == 0:
        s_t.append(to_state(f))
        f_pre = deepcopy(f)
        f = SL.SudoSolveOneStep(f, [], 1)
        fill = f - f_pre
        a_t.append(to_action(fill))
        s_t_plus_one.append(to_state(f))
    scope.append(len(s_t))

s_t = np.stack(s_t)
a_t = np.stack(a_t)
s_t_plus_one = np.stack(s_t_plus_one)
scope = np.stack(scope)

torch.save(torch.tensor(s_t), os.path.join(data_set+difficulty, difficulty+"_st.pt"))
torch.save(torch.tensor(a_t), os.path.join(data_set+difficulty, difficulty+"_at.pt"))
torch.save(torch.tensor(s_t_plus_one), os.path.join(data_set+difficulty, difficulty+"_st_plusOne.pt"))
torch.save(torch.tensor(scope), os.path.join(data_set+difficulty, difficulty+"_scope.pt"))

print(s_t.shape)
print(s_t_plus_one.shape)







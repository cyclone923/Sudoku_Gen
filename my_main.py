from src.burnImage import burnSudo2Img as BS
from src.generators import Generators as Gen
from src.utils import GeneratorUtils as GU
from src.ratingSudos import rating as RT
from src.checkers import SudoCheck as SC
from src.solver import solver as SL

import numpy as np
import os
from collections import defaultdict


def onPressSolve(A):
    if (np.min(A) >= 0 and np.max(A) <= 9):
        if (SC.IsSudoRight(A)):
            print("Solver", "Already Solved!")
        else:
            Adef = A.copy()
            if (GU.CountSolutions(Adef, [], 0, 1) == 1):
                sol_A = SL.SudoSolveIt2(A, [], 1)
                print(A)
            else:
                print("Solver", "There isn't unique solution!")
    else:
        print("Solver", "Wrong inputs!")

def onPressRate(A):
    if (np.min(A) >= 0 and np.max(A) <= 9):
        if (SC.IsSudoRight(A)):
            print("Rating", "Already Solved!")
        else:
            Adef = A.copy()
            if (GU.CountSolutions(Adef, [], 0, 1) == 1):
                difficulty = RT.RateProb(A)
                # print("Rating", "Your Problem is " + BS.getStrDiff(difficulty))
                return difficulty
            else:
                print("Rating", "There isn't unique solution!")
                print(A)
                exit(0)
    else:
        print("Rating", "Wrong inputs!")

def rateBatch(A_batch):
    diff_dict = defaultdict(lambda : 0)
    known_dict = defaultdict(lambda : 0)
    for i,A in enumerate(A_batch):
        # difficulty = onPressRate(A)
        # diff_dict[difficulty] += 1
        # print(np.count_nonzero(A))
        known_dict[np.count_nonzero(A)] += 1
    print(diff_dict)
    print(known_dict)
    print(max(known_dict.keys()), min(known_dict.keys()))



if __name__ == "__main__":
    data_dir = "sudoku"
    # A = np.array([[0,0,0,0,0,2,0,4,7],
    #               [0,7,0,0,0,0,0,0,9],
    #               [0,0,0,0,4,0,0,0,0],
    #               [0,1,0,0,6,5,4,0,0],
    #               [0,8,0,0,0,0,6,0,0],
    #               [0,0,0,4,0,9,0,2,5],
    #               [0,0,2,0,5,4,0,0,0],
    #               [8,0,0,0,0,6,0,0,0],
    #               [0,0,3,9,0,0,1,0,0]])
    # onPressSolve(A)
    # exit(0)
    features = np.load(os.path.join(data_dir, 'features.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    rateBatch(features)
from copy import deepcopy
import numpy as np
from src.utils import SolverUtils as SV
from src.checkers import SudoCheck as SC

def multi_rule_inference(C):
    C, er3 = SV.CandLineEr(C)
    if er3:
        return C, True
    C, er4 = SV.multLineEr(C)
    if er4:
        return C, True
    C, er5 = SV.nakedPairEr(C)
    if er5:
        return C, True
    C, er6 = SV.nakedTuplesEr(C)
    if er6:
        return C, True
    C, er7 = SV.hiddenPairEr(C)
    if er7:
        return C, True
    C, er8 = SV.hiddenTupleEr(C)
    if er8:
        return C, True
    C, er9 = SV.XWingEr(C)
    if er9:
        return C, True
    C, er10 = SV.SwordFishEr(C)
    if er10:
        return C, True
    return C, False



def SudoSolveOneStep(A, C, n):
    if n == 1:
        C = SV.ConstructC(A)
        return SudoSolveOneStep(A, C, 2)
    else:
        while True:
            A, C, DidIn1 = SV.SudoInput1(A, C, one_step=True)
            if DidIn1:
                return A
            A, C, DidIn2 = SV.SudoInput2(A, C, one_step=True)
            if DidIn2:
                return A
            C, DidInfer = multi_rule_inference(C)
            if DidInfer:
                continue
            else:
                r = SV.FindMinRow(C)
                rC = C[r, :]
                for i in range(9):
                    if (rC[i] > 0):
                        A2 = deepcopy(A)
                        C2 = deepcopy(C)
                        A2[rC[9], rC[10]] = rC[i]
                        A2_before = deepcopy(A2)
                        C2 = SV.clearC(C2, rC[9], rC[10], rC[11], rC[i])
                        C2 = np.delete(C2, r, 0)
                        A2 = SudoSolveIt2(A2, C2, 2)
                        if np.min(A2) > 0:
                            return A2_before


def SudoSolveIt(A,C,n):
    if n==1:
        C=SV.ConstructC(A)
        return SudoSolveIt(A,C,2)
    else:
        if np.min(A)>0:
            return A
        C, t2 = SV.SwordFishEr(C)
        if(t2):
            print("done")
        A, C, DidIn = SV.SudoInput1(A,C)
        A, C, DidIn = SV.SudoInput2(A,C)
        C, t2 = SV.SwordFishEr(C)
        #C, t = SV.XWingEr(C)
        #print(C)
        
        C, t = SV.CandLineEr(C)
        #C, t = SV.XWingEr(C)
        C, t = SV.multLineEr(C)
        #C, t2 = SV.SwordFishEr(C)
        #C, t = SV.nakedPairEr(C)
        #C, t = SV.nakedTuplesEr(C)
        C, t = SV.hiddenPairEr(C)
        #C, t = SV.hiddenTupleEr(C)
        #C, t2 = SV.SwordFishEr(C)
        
        if(t2):
            print("done")
        
        return SudoSolveIt(A,C,2)


def SudoSolveIt1(A,C,n):
    Arec = deepcopy(A)
    Crec = deepcopy(C)
    if n==1:
        Crec=SV.ConstructC(Arec)
        return SudoSolveIt1(Arec,Crec,2)
    else:
        if (n==2):
            Arec, Crec, DidIn = SV.SudoInput1(Arec,Crec)
            return SudoSolveIt1(Arec,Crec,3)
        if np.min(Arec)>0:
            return Arec
        if (SC.IsDeadEnd(Crec)):
            return np.array([0,0])
        Arec, Crec, DidIn = SV.SudoInput1(Arec,Crec)
        if (DidIn):
            return SudoSolveIt1(Arec,Crec,3)
        else:
            r=SV.FindMinRow(Crec)
            rC=Crec[r,:]
            for i in range(9):
                if (rC[i]>0):
                    A2=deepcopy(Arec)
                    C2=deepcopy(Crec)
                    A2[rC[9],rC[10]]=rC[i]
                    C2=SV.clearC(C2,rC[9],rC[10],rC[11],rC[i])
                    C2=np.delete(C2,r,0)
                    A2=SudoSolveIt1(A2,C2,3)
                    if np.min(A2)>0:
                        return A2
            return A2


def SudoSolveIt2(A,C,n):    #Fastest Solver
    Arec = A
    Crec = C
    if n==1:
        Crec=SV.ConstructC(Arec)
        return SudoSolveIt2(Arec,Crec,2)
    else:
        if (n==2):
            Arec, Crec, DidIn = SV.SudoInput1(Arec,Crec)
            Arec, Crec, DidIn = SV.SudoInput2(Arec,Crec)
            return SudoSolveIt2(Arec,Crec,3)
        if np.min(Arec)>0:
            return Arec
        if (SC.IsDeadEnd(Crec)):
            return np.array([0,0])
        Arec, Crec, DidIn = SV.SudoInput1(Arec,Crec)
        if (DidIn):
            if (SC.IsDeadEnd(Crec)):
                return np.array([0,0])
            else:
                Arec, Crec, DidIn = SV.SudoInput2(Arec,Crec)
                return SudoSolveIt2(Arec,Crec,3)
        else:
            Arec, Crec, DidIn = SV.SudoInput2(Arec,Crec)
            if (DidIn):
                return SudoSolveIt2(Arec,Crec,3)
        r=SV.FindMinRow(Crec)
        rC=Crec[r,:]
        for i in range(9):
            if (rC[i]>0):
                A2=deepcopy(Arec)
                C2=deepcopy(Crec)
                A2[rC[9],rC[10]]=rC[i]
                C2=SV.clearC(C2,rC[9],rC[10],rC[11],rC[i])
                C2=np.delete(C2,r,0)
                A2=SudoSolveIt2(A2,C2,3)
                if np.min(A2)>0:
                    return A2
        return A2


def SudoBruteSolve(A,n):    #Brute Force solver
    if (SC.IsSudoRight(A)):
        return A
    Arec=deepcopy(A)
    for i in range(9):
        for j in range(9):
            if (Arec[i,j]==0):
                posnum=SV.FindPosNums(Arec,i,j)
                if (posnum.size>0):
                    for k in range(posnum.size):
                        Arec[i,j]=posnum[k]
                        A2=SudoBruteSolve(Arec,n)
                        if (SC.IsSudoRight(A2)):
                            return A2
                    return A2
                else:
                    return Arec
                    
                


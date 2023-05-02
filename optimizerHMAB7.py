import MAB.HMAB7 as bandit



import numpy as np 
import time

import os



def MBArun(objf, lb, ub, dim, NumOfRuns, popSize,Iter):


    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    objectivefunc : function
        problem - benchmark functions
    lb   : list
    different dimensions from [-5,5]^n, n belongs to {2,3,4,5,10,20,40}
    
    ub   : list 
    different dimensions from [-5,5]^n, n belongs to {2,3,4,5,10,20,40}
    
    NumOfRuns : int
        The number of independent runs
    popSize  : int
        Size of population
    Iter     : int
        The number of iterations
    Returns
    -----------
    N/A
    """
           
    for k in range(0, NumOfRuns):
        convergence = [0] * NumOfRuns
        executionTime = [0] * NumOfRuns
        stopiter = [0] * NumOfRuns
        model= bandit.BANDIT(objf, lb, ub, dim, popSize, Iter)
        x = model.optimize()       
        convergence[k] = x.convergence


        
    


   

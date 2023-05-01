import MAB.HMAB7 as bandit



import numpy as np 
import time

import os



def MBArun(objf, lb, ub, dim, NumOfRuns, popSize,Iter):


    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)

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


        
    


   

import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from solution import solution

class BANDIT():
    def __init__(self, objf, lb, ub, dim, SearchAgents_no, Max_iter):
        super(BANDIT, self).__init__()
        self.objf = objf
        self.s = solution()
        self.s.optimizer = "BANDIT"
        #self.s.objfname = self.objf.__name__

        self.s.executionTime = 0
        self.s.R_list = []
        self.best_pos = np.zeros(dim)
        self.best_sol = float("inf")  # change this to -inf for maximization problems
        self.dim = dim
        self.lb= lb
        self.ub= ub
        self.SearchAgents_no = SearchAgents_no
        self.Max_iter = Max_iter
        self.recent = []

        

        # Initialize the locations of Harris' hawks
        self.Positions = np.asarray(
            [x * (self.ub - self.lb) + self.lb for x in np.random.uniform(0, 1, (self.SearchAgents_no, self.dim))]
        )
        self.fitness = np.full(self.SearchAgents_no, np.inf)
        # Initialize convergence
        self.convergence_curve = np.zeros(self.Max_iter)
        ############################
        
        ##print('BANDIT is now tackling  "' + self.objf.__name__ + '"')
        self.timerStart = time.time()
        self.s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        ############################

        self.t = 0  # Loop counter
        self.f = 0 ### fitness evaluation
        self.ratio = 1
        self.K = 50 # available arms in selections
        ##################################################################
        self.c = 1
        self.Q = np.zeros(8)
        self.eps = 1e-6

        self.R =[]
        self.exp_R= []
        self.count = np.zeros_like(self.Q)  # N(a)=0                         
        self.reward_list = []       # List of rewards
        self.reward_avg_list = []    # List of averaged rewards
        self.begin = True 
        
    def optimize(self):

        if self.t == 0:
            for i in range(0, self.SearchAgents_no):
                # Check boundries
                self.Positions[i, :] = np.clip(self.Positions[i, :], self.lb, self.ub)
                # fitness of locations
                self.fitness[i] = self.objf(self.Positions[i, :])
                # Update the location of Rabbit
                if self.fitness[i] < self.best_sol:  # Change this to > for maximization problem
                    self.best_sol = self.fitness[i].copy()
                    self.best_pos = self.Positions[i, :].copy()
            self.convergence_curve[self.t] = self.best_sol
            self.t = self.t + 1
        
        if self.t < self.K * len(self.Q) + 1:     # Take each action once 
            self.begin = True 
            #if self.t < self.K+1:
            if self.t < self.K+1:
                self.count[1] +=1
                self.DE()
            elif self.t < 2 * self.K+1:
                self.count[0] +=1
                self.HHO() 
            elif self.t < 3 * self.K+1:
                self.count[2] +=1
                self.WOA()
            elif self.t < 4 * self.K+1:
                self.count[3] +=1
                self.PSO()
            elif self.t < 5 * self.K+1:
                self.count[4] +=1
                self.JAYA()
            elif self.t < 6 * self.K+1:
                self.count[5] +=1
                self.FFA()
            elif self.t < 7 * self.K+1:
                self.count[6] +=1
                self.BAT()
            #elif self.t < 8 * self.K+1:
                #self.count[8] +=1
                #self.CS()
            #elif self.t < 8 * self.K+1:
                #self.count[7] +=1
                #self.GA()

            
        else:
            self.begin = False
            term = np.sqrt((np.log(self.t))/self.count)
            ucb = self.Q+self.c*term
            ind = np.random.choice(np.where(ucb == ucb.max())[0])  # break ties randomly
            if ind == 0:
                self.HHO()
            elif ind == 1:
                self.DE()
            elif ind == 2:
                self.WOA()
            elif ind == 3:
                self.PSO()
            elif ind == 4:
                self.JAYA()
            elif ind == 5:
                self.FFA()
            elif ind == 6:
                self.BAT()
            #elif ind == 7:
                #self.CS()
            #elif ind == 7:
                #self.GA()         
        return self.s

    def soft(self, ucb, index):
        ucb = ucb - np.max(ucb)
        return np.exp(ucb[index]) / (np.sum(np.exp(ucb)) )
    
    ######################################################################################################### HHO 

    def Levy(self):
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = 0.01 * np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        zz = np.power(np.absolute(v), (1 / beta))
        step = np.divide(u, zz)
        return step

    def HHO(self):
        # Main loop
        #print('HHO is optimizing  "' + self.objf.__name__ + '"')
        while self.t < self.Max_iter:
            E1 = 2 * (1 - (self.t / self.Max_iter))  # factor to show the decreaing energy of rabbit

            # Update the location of Harris' hawks
            for i in range(0, self.SearchAgents_no):
                E0 = 2 * random.random() - 1  # -1<E0<1
                Escaping_Energy = E1 * (
                    E0
                )  # escaping energy of rabbit Eq. (3) in the paper
                # -------- Exploration phase Eq. (1) in paper -------------------
                if abs(Escaping_Energy) >= 1:
                    # Harris' hawks perch randomly based on 2 strategy:
                    q = random.random()
                    rand_Hawk_index = math.floor(self.SearchAgents_no * random.random())
                    Positions_rand = self.Positions[rand_Hawk_index, :]
                    if q < 0.5:
                        # perch based on other family members
                        self.Positions[i, :] = Positions_rand - random.random() * abs(
                            Positions_rand - 2 * random.random() * self.Positions[i, :]
                        )
                    elif q >= 0.5:
                        # perch on a random tall tree (random site inside group's home range)
                        self.Positions[i, :] = (self.best_pos - self.Positions.mean(0)) - random.random() * (
                            (self.ub - self.lb) * random.random() + self.lb
                        )
                # -------- Exploitation phase -------------------
                elif abs(Escaping_Energy) < 1:
                    # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    # phase 1: ----- surprise pounce (seven kills) ----------
                    # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r = random.random()  # probablity of each event

                    if (
                        r >= 0.5 and abs(Escaping_Energy) < 0.5
                    ):  # Hard besiege Eq. (6) in paper
                        self.Positions[i, :] = (self.best_pos) - Escaping_Energy * abs(
                            self.best_pos - self.Positions[i, :]
                        )

                    if (
                        r >= 0.5 and abs(Escaping_Energy) >= 0.5
                    ):  # Soft besiege Eq. (4) in paper
                        Jump_strength = 2 * (
                            1 - random.random()
                        )  # random jump strength of the rabbit
                        self.Positions[i, :] = (self.best_pos - self.Positions[i, :]) - Escaping_Energy * abs(
                            Jump_strength * self.best_pos - self.Positions[i, :]
                        )

                    # phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if (
                        r < 0.5 and abs(Escaping_Energy) >= 0.5
                    ):  # Soft besiege Eq. (10) in paper
                        # rabbit try to escape by many zigzag deceptive motions
                        Jump_strength = 2 * (1 - random.random())
                        Positions1 = self.best_pos - Escaping_Energy * abs(
                            Jump_strength * self.best_pos - self.Positions[i, :]
                        )
                        Positions1 = np.clip(Positions1, self.lb, self.ub)

                        if self.objf(Positions1) < self.fitness[i]:  # improved move?
                            self.Positions[i, :] = Positions1.copy()
                        else:  # hawks perform levy-based short rapid dives around the rabbit
                            Positions2 = (
                                self.best_pos
                                - Escaping_Energy
                                * abs(Jump_strength * self.best_pos - self.Positions[i, :])
                                + np.multiply(np.random.randn(self.dim), self.Levy())
                            )
                            Positions2 = np.clip(Positions2, self.lb, self.ub)
                            if self.objf(Positions2) < self.fitness[i]:
                                self.Positions[i, :] = Positions2.copy()
                    if (
                        r < 0.5 and abs(Escaping_Energy) < 0.5
                    ):  # Hard besiege Eq. (11) in paper
                        Jump_strength = 2 * (1 - random.random())
                        Positions1 = self.best_pos - Escaping_Energy * abs(
                            Jump_strength * self.best_pos - self.Positions.mean(0)
                        )
                        Positions1 = np.clip(Positions1, self.lb, self.ub)

                        if self.objf(Positions1) < self.fitness[i]:  # improved move?
                            self.Positions[i, :] = Positions1.copy()
                        else:  # Perform levy-based short rapid dives around the rabbit
                            Positions2 = (
                                self.best_pos
                                - Escaping_Energy
                                * abs(Jump_strength * self.best_pos - self.Positions.mean(0))
                                + np.multiply(np.random.randn(self.dim), self.Levy())
                            )
                            Positions2 = np.clip(Positions2, self.lb, self.ub)
                            if self.objf(Positions2) < self.fitness[i]:
                                self.Positions[i, :] = Positions2.copy()

            for i in range(0, self.SearchAgents_no):
                # Check boundries
                self.Positions[i, :] = np.clip(self.Positions[i, :], self.lb, self.ub)
                # fitness of locations
                self.fitness[i] = self.objf(self.Positions[i, :])
                # Update the location of Rabbit
                if self.fitness[i] < self.best_sol:  # Change this to > for maximization problem
                    self.best_sol = self.fitness[i].copy()
                    self.best_pos = self.Positions[i, :].copy()

            self.f += self.SearchAgents_no
            self.convergence_curve[self.t] = self.best_sol
            if self.t % 1 == 0:
                print(
                    [
                        "At iteration "
                        + str(self.t +1)
                        + " the best fitness is "
                        + str(self.best_sol)
                    ]
                )
            self.t = self.t + 1
            self.s.R_list.append(0)

            if (self.begin == True):
                # if self.t != 1:
                # reward = self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1] ## Original Reward 
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps) ## Revised Reward 
                # self.Q[0] = self.Q[0] + (1/self.count[0])*(reward-self.Q[0])
                # self.Q[0] = reward
                self.Q[0] = self.ratio * reward + (1-self.ratio) * self.Q[0] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (HHO): ", self.Q)
                self.recent.append(reward)
                self.optimize()
            
            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[0] = self.ratio * reward + (1-self.ratio) * self.Q[0] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (HHO): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.random.choice(7, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6),self.soft(ucb,7)])

                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])
                self.count[ind] +=1
                if ind == 1:
                    self.DE()
                elif ind ==2:
                    self.WOA()
                elif ind ==3:
                    self.PSO()
                elif ind ==4:
                    self.JAYA()
                elif ind ==5:
                    self.FFA()
                elif ind ==6:
                    self.BAT()
                #elif ind ==7:
                    #self.CS()
                #elif ind ==7:
                    #self.GA()

        timerEnd = time.time()
        self.s.stopiter = self.f
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve

    ######################################################################################################### DE
    def DE(self):
        mutation_factor = 0.5
        crossover_ratio = 0.7
        #print('DE is optimizing  "' + self.objf.__name__ + '"')
        while self.t < self.Max_iter:
            # loop through population
            for i in range(self.SearchAgents_no):
                # 1. Mutation
                # select 3 random solution except current solution
                ids_except_current = [_ for _ in range(self.SearchAgents_no) if _ != i]
                id_1, id_2, id_3 = random.sample(ids_except_current, 3)
                mutant_sol = []
                for d in range(self.dim):
                    d_val = self.Positions[id_1, d] + mutation_factor * (
                        self.Positions[id_2, d] - self.Positions[id_3, d]
                    )
                    # 2. Recombination
                    rn = random.uniform(0, 1)
                    if rn > crossover_ratio:
                        d_val = self.Positions[i, d]
                    # add dimension value to the mutant solution
                    mutant_sol.append(d_val)
                # 3. Replacement / Evaluation
                # clip new solution (mutant)
                mutant_sol = np.clip(mutant_sol, self.lb, self.ub)
                # calc fitness
                mutant_fitness = self.objf(mutant_sol)
                if mutant_fitness < self.fitness[i]:
                    self.Positions[i, :] = mutant_sol
                    self.fitness[i] = mutant_fitness
                    # update leader
                    if mutant_fitness < self.best_sol:
                        self.best_sol = mutant_fitness
                        self.best_pos = mutant_sol

            self.f += self.SearchAgents_no
            self.convergence_curve[self.t] = self.best_sol
            if self.t % 1 == 0:
                print(
                    ["At iteration " + str(self.t + 1) + " the best fitness is " + str(self.best_sol)]
                )
            # increase iterations
            self.t = self.t + 1
            self.s.R_list.append(1)

            if self.begin == True:
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps ) ## Revised Reward 
                self.Q[1] = self.ratio * reward + (1-self.ratio) * self.Q[1] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (DE): ", self.Q)
                self.recent.append(reward)
                self.optimize()
      
            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[1] = self.ratio * reward + (1-self.ratio) * self.Q[1]  
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (DE): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.random.choice(8, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb, 3),self.soft(ucb, 4),self.soft(ucb, 5),self.soft(ucb, 6),self.soft(ucb, 7)])
                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])

                self.count[ind] +=1
                if ind == 0:
                    self.HHO()
                elif ind == 2:
                    self.WOA()
                elif ind ==3:
                    self.PSO()
                elif ind ==4:
                    self.JAYA()
                elif ind ==5:
                    self.FFA()
                elif ind ==6:
                    self.BAT()
                #elif ind ==7:
                    #self.CS()
                #elif ind ==7:
                    #self.GA()
                

        timerEnd = time.time()
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve

    ######################################################################################################### WOA
    def WOA(self):
        #print('WOA is optimizing  "' + self.objf.__name__ + '"')
        # Main loop
        while self.t < self.Max_iter:
            a = 2 - self.t * ((2) / self.Max_iter)
            # a decreases linearly fron 2 to 0 in Eq. (2.3)
            # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a2 = -1 + self.t * ((-1) / self.Max_iter)

            # Update the Position of search agents
            for i in range(0, self.SearchAgents_no):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]
                A = 2 * a * r1 - a  # Eq. (2.3) in the paper
                C = 2 * r2  # Eq. (2.4) in the paper
                b = 1
                #  parameters in Eq. (2.5)
                l = (a2 - 1) * random.random() + 1  #  parameters in Eq. (2.5)
                p = random.random()  # p in Eq. (2.6)
                for j in range(0, self.dim):
                    if p < 0.5:
                        if abs(A) >= 1:
                            rand_leader_index = math.floor(
                                self.SearchAgents_no * random.random()
                            )
                            X_rand = self.Positions[rand_leader_index, :]
                            D_X_rand = abs(C * X_rand[j] - self.Positions[i, j])
                            self.Positions[i, j] = X_rand[j] - A * D_X_rand
                        elif abs(A) < 1:
                            D_Leader = abs(C * self.best_pos[j] - self.Positions[i, j])
                            self.Positions[i, j] = self.best_pos[j] - A * D_Leader
                    elif p >= 0.5:

                        distance2Leader = abs(self.best_pos[j] - self.Positions[i, j])
                        # Eq. (2.5)
                        self.Positions[i, j] = (
                            distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                            + self.best_pos[j]
                        )
            for i in range(0, self.SearchAgents_no):
                # Return back the search agents that go beyond the boundaries of the search space
                # Positions[i,:]=checkBounds(Positions[i,:],lb,ub)
                self.Positions[i,:] = np.clip(self.Positions[i, :], self.lb , self.ub)
                # Calculate objective function for each search agent
                self.fitness[i] = self.objf(self.Positions[i, :])
                # Update the leader
                if self.fitness[i] < self.best_sol:  # Change this to > for maximization problem
                    self.best_sol = self.fitness[i]
                    # Update alpha
                    self.best_pos = self.Positions[i, :].copy()  # copy current whale position into the leader position

            self.convergence_curve[self.t] = self.best_sol
            if self.t % 1 == 0:
                print(
                    ["At iteration " + str(self.t +1) + " the best fitness is " + str(self.best_sol)]
                )
            self.t = self.t + 1
            self.s.R_list.append(2)

            if self.begin == True:
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps ) ## Revised Reward 
                self.Q[2] = self.ratio * reward + (1-self.ratio) * self.Q[2]  
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (WOA): ", self.Q)
                self.recent.append(reward)
                self.optimize()

            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[2] = self.ratio * reward + (1-self.ratio) * self.Q[2] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (WOA): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.random.choice(8, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb, 6),self.soft(ucb, 7)])
                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])
                self.count[ind] +=1
                if ind == 1:
                    self.DE()
                elif ind == 0:
                    self.HHO()
                elif ind == 3:
                    self.PSO()
                elif ind == 4:
                    self.JAYA()
                elif ind == 5:
                    self.FFA()
                elif ind == 6:
                    self.BAT()
                #elif ind ==7:
                    #self.CS()
                elif ind == 7:
                    self.GA()
                

        timerEnd = time.time()
        #self.s.stopiter = self.f
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve

    ######################################################################################################### PSO
    def PSO(self):
        # PSO parameters

        Vmax = 6
        wMax = 0.9
        wMin = 0.2
        c1 = 2
        c2 = 2
        
        #if function of PSO can run,you do not use it or not
        vel = np.zeros((self.SearchAgents_no, self.dim))

        pBestScore = np.zeros(self.SearchAgents_no)
        pBestScore.fill(float("inf"))

        pBest = np.zeros((self.SearchAgents_no, self.dim))
        
        ############################################
        #print('PSO is optimizing  "' + self.objf.__name__ + '"')

        
        
        
        for self.t in range(0, self.Max_iter):
            for i in range(0, self.SearchAgents_no):
                # pos[i,:]=checkBounds(pos[i,:],lb,ub)
                for j in range(self.dim):
                    self.Positions[i, j] = np.clip(self.Positions[i, j], self.lb[j], self.ub[j])
                # Calculate objective function for each particle
                self.fitness = self.objf(self.Positions[i, :])

                if pBestScore[i] > self.fitness:
                    pBestScore[i] = self.fitness
                    pBest[i, :] = self.Positions[i, :].copy()

                if self.best_sol > self.fitness:
                    self.best_sol = self.fitness
                    self.best_pos = self.Positions[i, :].copy()

            # Update the W of PSO
            w = wMax - self.t * ((wMax - wMin) / self.Max_iter)

            for i in range(0, self.SearchAgents_no):
                for j in range(0, self.dim):
                    r1 = random.random()
                    r2 = random.random()
                    vel[i, j] = (
                        w * vel[i, j]
                        + c1 * r1 * (pBest[i, j] - self.Positions[i, j])
                        + c2 * r2 * (self.best_pos[j] - self.Positions[i, j])
                     )

                    if vel[i, j] > Vmax:
                        vel[i, j] = Vmax

                    if vel[i, j] < -Vmax:
                        vel[i, j] = -Vmax

                    self.Positions[i, j] = self.Positions[i, j] + vel[i, j]

            self.convergence_curve[self.t] = self.best_sol

            if self.t % 1 == 0:
                print(
                    ["At iteration " + str(self.t + 1) + " the best fitness is " + str(self.best_sol)]
                 )        
        
            self.t = self.t + 1
            self.s.R_list.append(3)

            if self.begin == True:
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps ) ## Revised Reward 
                self.Q[3] = self.ratio * reward + (1-self.ratio) * self.Q[3]  
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (PSO): ", self.Q)
                self.recent.append(reward)
                self.optimize()

            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[3] = self.ratio * reward + (1-self.ratio) * self.Q[3] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (PSO): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.random.choice(8, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6),self.soft(ucb,7)])
                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])
                self.count[ind] +=1
                if ind == 0:
                    self.HHO()
                elif ind == 1:
                    self.DE()
                elif ind == 2:
                    self.WOA()
                elif ind == 4:
                    self.JAYA()
                elif ind == 5:
                    self.FFA()
                elif ind == 6:
                    self.BAT()
                #elif ind ==7:
                    #self.CS()
                #elif ind == 7:
                    #self.GA()
                
    
                     
        timerEnd = time.time()
        self.s.stopiter = self.f
        self.s.executionTime = timerEnd - timerStart
        self.s.convergence = self.convergence_curve



    ######################################################################################################### JAYA
    def JAYA(self):


        fitness_matrix = np.zeros((self.SearchAgents_no))


        for i in range(0, self.SearchAgents_no):

            fitness_matrix[i] = self.fitness[i]


        # Main loop
        for self.t in range(0, self.Max_iter):

            # Update the Position of search agents
            for i in range(0, self.SearchAgents_no):
                New_Position = np.zeros(self.dim)
                for j in range(0, self.dim):

                    # Update r1, r2
                    r1 = random.random()
                    r2 = random.random()

                    # JAYA Equation
                    New_Position[j] = (
                        self.Positions[i][j]
                        + r1 * (self.best_pos[j] - abs(self.Positions[i, j]))
                        - r2 * (self.worst_pos[j] - abs(self.Positions[i, j]))
                    )

                    # checking if New_Position[j] lies in search space
                    if New_Position[j] > self.ub[j]:
                        New_Position[j] = self.ub[j]
                    if New_Position[j] < self.lb[j]:
                        New_Position[j] = self.lb[j]

                new_fitness = self.objf(New_Position)
                current_fit = fitness_matrix[i]

                # replacing current element with new element if it has better fitness
                if new_fitness < current_fit:
                    self.Positions[i] = New_Position
                    fitness_matrix[i] = new_fitness

            # finding the best and worst element
            for i in range(self.SearchAgents_no):
                if fitness_matrix[i] < self.best_sol:
                    self.best_sol = fitness_matrix[i]
                    self.best_pos = self.Positions[i, :].copy()

                if fitness_matrix[i] > self.worst_score:
                    self.worst_score = fitness_matrix[i]
                    self.worst_pos = self.Positions[i, :].copy()
 
            self.convergence_curve[self.t] = self.best_sol


            if self.t % 1 == 0:
                print(
                    ["At iteration " + str(self.t +1) + " the best fitness is " + str(self.best_sol)]
                )
                
            
            
            self.t = self.t + 1
            self.s.R_list.append(4)

            if self.begin == True:
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps ) ## Revised Reward 
                self.Q[4] = self.ratio * reward + (1-self.ratio) * self.Q[4]  
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (JAYA): ", self.Q)
                self.recent.append(reward)
                self.optimize()

            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[4] = self.ratio * reward + (1-self.ratio) * self.Q[4] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (JAYA): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.random.choice(8, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb, 6),self.soft(ucb, 7)])
                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])
                self.count[ind] +=1
                if ind == 3:
                    self.PSO()
                elif ind == 2:
                    self.WOA()              
                elif ind == 1:
                    self.DE()               
                elif ind == 0:
                    self.HHO()              
                elif ind == 5:
                    self.FFA()
                elif ind == 6:
                    self.BAT()
                #elif ind ==7:
                    #self.CS()
                #elif ind == 7:
                    #self.GA()
               
        
        
        timerEnd = time.time()
        self.s.stopiter = self.f
        s.executionTime = timerEnd - timerStart
        self.s.convergence = self.convergence_curve



    
    ######################################################################################################### FFA
    
    def alpha_new(alpha, NGen):
        #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
        #% alpha_0=0.9
        delta = 1 - (10 ** (-4) / 0.9) ** (1 / NGen)
        alpha = (1 - delta) * alpha
        return alpha



    def FFA(self):

        # General parameters

        # n=50 #number of fireflies
        # dim=30 #dim
        # lb=-50
        # ub=50
        # MaxGeneration=500

        # FFA parameters
        alpha = 0.5  # Randomness 0--1 (highly random)
        betamin = 0.20  # minimum value of beta
        gamma = 1  # Absorption coefficient



        zn = np.ones(self.SearchAgents_no)*np.inf
        zn.fill(float("inf"))
        convergence =[]


        #print('FFA is optimizing  "' + self.objf.__name__ + '"')


        # Main loop
        for self.t in range(0, self.Max_iter):  # start iterations

            #% This line of reducing alpha is optional
            
            alpha = (1 - (1 - (10 ** (-4) / 0.9) ** (1 / self.Max_iter))) * alpha

            #% Evaluate new solutions (for all n fireflies)
            for i in range(0, self.SearchAgents_no):
                zn[i] = self.objf(self.Positions[i, :])
                self.fitness [i] = zn[i]
                

            # Ranking fireflies by their light intensity/objectives

            self.fitness = np.sort(zn)
            Index = np.argsort(zn)
            self.Positions = self.Positions[Index, :]

            # Find the current best
            nso = self.Positions
            Lighto = self.fitness 
            self.best_pos = self.Positions[0, :]
            Lightbest = self.fitness [0]

            #% For output only
            self.best_sol = Lightbest

            #% Move all fireflies to the better locations
            #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
            #          Lightbest,alpha,betamin,gamma,Lb,Ub);
            scale = []
            for b in range(self.dim):
                scale.append(abs(self.ub[b] - self.lb[b]))
            scale = np.array(scale)
            for i in range(0, self.SearchAgents_no):
                # The attractiveness parameter beta=exp(-gamma*r)
                for j in range(0, self.SearchAgents_no):
                    r = np.sqrt(np.sum((self.Positions[i, :] - self.Positions[j, :]) ** 2))
                    # r=1
                    # Update moves
                    if self.fitness[i] > Lighto[j]:  # Brighter and more attractive
                        beta0 = 1
                        beta = (beta0 - betamin) * math.exp(-gamma * r ** 2) + betamin
                        tmpf = alpha * (np.random.rand(self.dim) - 0.5) * scale
                        self.Positions[i, :] = self.Positions[i, :] * (1 - beta) + nso[j, :] * beta + tmpf
    

            convergence.append(self.best_sol)
            self.convergence_curve[self.t] = self.best_sol
            #self.f += self.SearchAgents_no

            #IterationNumber = self.t
            BestQuality = self.best_sol

            if self.t % 1 == 0:
                print(
                    ["At iteration " + str(self.t +1) + " the best fitness is " + str(self.best_sol)]
                )
            
        
            self.t = self.t + 1
            self.s.R_list.append(0)

            if (self.begin == True):
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps) ## Revised Reward 
                self.Q[5] = self.ratio * reward + (1-self.ratio) * self.Q[5] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (FFA): ", self.Q)
                self.recent.append(reward)
                self.optimize()
            
            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[5] = self.ratio * reward + (1-self.ratio) * self.Q[5] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (FFA): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.random.choice(8, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6),self.soft(ucb,7)])
                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])

                self.count[ind] +=1
                if ind == 0:
                    self.HHO()
                elif ind == 1:
                    self.DE()
                elif ind == 2:
                    self.WOA()
                elif ind == 3:
                    self.PSO()
                elif ind == 4:
                    self.JAYA()               
                elif ind == 6:
                    self.BAT()
                #elif ind ==7:
                    #self.CS()
                #elif ind == 7:
                    #self.GA()


        timerEnd = time.time()
        self.s.stopiter = self.f
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve

     
                    
    ######################################################################################################### BAT
    def BAT(self):

        ##n = N
        # Population size

        ##if not isinstance(lb, list):
        ##lb = [lb] * dim
        ##if not isinstance(ub, list):
        ##ub = [ub] * dim
        ##N_gen = Max_iteration  # Number of generations

        A = 0.5
        # Loudness  (constant or decreasing)
        r = 0.5
        # Pulse rate (constant or decreasing)

        Qmin = 0  # Frequency minimum
        Qmax = 2  # Frequency maximum

        ##d = dim  # Number of dimensions

        # Initializing arrays
        Q = np.zeros(self.SearchAgents_no)  # Frequency
        v = np.zeros((self.SearchAgents_no, self.dim))  # Velocities
        ##Convergence_curve = []

        # Initialize the population/solutions
        S = np.zeros((self.SearchAgents_no, self.dim))
        S = np.copy(self.Positions)
        

        # Find the initial best solution and minimum fitness
        #It's better to use np.argmin to find the minmum fitness, but we need to update it to follow initialization 
        #I = np.argmin(self.fitness)
        #self.best_pos = Sol[I, :]
        #self.best_sol = min(Fitness)
         
        
        # Main loop
        for self.t in range(0, self.Max_iter):

            # Loop over all bats(solutions)
            for i in range(0, self.SearchAgents_no):
                Q[i] = Qmin + (Qmin - Qmax) * random.random()
                #v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
                v[i, :] = v[i, :] + (self.Positions[i, :] - self.best_pos) * Q[i]
                S[i, :] = self.Positions[i, :] + v[i, :]

                # Check boundaries
                for j in range(self.dim):
                    self.Positions[i, j] = np.clip(self.Positions[i, j], self.lb[j], self.ub[j])

                # Pulse rate
                if random.random() > r:
                    S[i, :] = self.best_pos + 0.001 * np.random.randn(self.dim)

                # Evaluate new solutions
                Fnew = self.objf(S[i, :])

                # Update if the solution improves
                if (Fnew <= self.fitness[i]) and (random.random() < A):
                    self.Positions[i, :] = np.copy(S[i, :])
                    self.fitness[i] = Fnew

                # Update the current best solution
                if Fnew <= self.best_sol:
                    self.best_pos = np.copy(S[i, :])
                    self.best_sol = Fnew 
            

            # update convergence curve
            self.f += self.SearchAgents_no
            self.convergence_curve[self.t] = self.best_sol
        
        

            if self.t % 1 == 0:
                #print(["At iteration " + str(self.t) + " the best fitness is " + str(fmin)])
                print(["At iteration " + str(self.t) + " the best fitness is " + str(self.best_sol)])
         
            self.t = self.t + 1
            self.s.R_list.append(2)

            if self.begin == True:
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps ) ## Revised Reward 
                self.Q[6] = self.ratio * reward + (1-self.ratio) * self.Q[6]  
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (BAT): ", self.Q)
                self.recent.append(reward)
                self.optimize()

            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[6] = self.ratio * reward + (1-self.ratio) * self.Q[6] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (BAT): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.random.choice(8, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb, 6),self.soft(ucb,7)])
                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])
                self.count[ind] +=1
                if ind == 1:
                    self.DE()
                elif ind == 0:
                    self.HHO()
                elif ind == 2:
                    self.WOA()
                elif ind == 3:
                    self.PSO()
                elif ind == 4:
                    self.JAYA()
                elif ind == 5:
                    self.FFA()
                #elif ind ==7:
                    #self.CS()
                #elif ind == 7:
                    #self.GA() 




        timerEnd = time.time()
        self.s.stopiter = self.f
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve




   

    ######################################################################################################### GA
    
    def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep):
        """
        The crossover of all individuals
        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        popSize: int
            Number of chrmosome in a population
        crossoverProbability: float
            The probability of crossing a pair of individuals
        keep: int
            Number of best individuals to keep without mutating for the next generation
        Returns
        -------
        N/A
        """
        # initialize a new population
        
        ##newPopulation = numpy.empty_like(population)
        ##newPopulation[0:keep] = population[0:keep]
        
        newPopulation = numpy.empty_like(self.Positions)
        newPopulation[0:keep] = self.Positions[0:keep]
        
        # Create pairs of parents. The number of pairs equals the number of individuals divided by 2
        for i in range(keep, self.SearchAgents_no, 2):
            # pair of parents selection
            ##parent1, parent2 = pairSelection(population, scores, popSize)
            parent1, parent2 = pairSelection(self.Positions, self.fitness, self.SearchAgents_no)        
            crossoverLength = min(len(parent1), len(parent2))
            parentsCrossoverProbability = random.uniform(0.0, 1.0)
            if parentsCrossoverProbability < crossoverProbability:
                offspring1, offspring2 = crossover(crossoverLength, parent1, parent2)
            else:
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()

            # Add offsprings to population
            newPopulation[i] = numpy.copy(offspring1)
            newPopulation[i + 1] = numpy.copy(offspring2)

        return newPopulation
    
    
    def mutatePopulaton(population, popSize, mutationProbability, keep, lb, ub):
        """
        The mutation of all individuals
        
        Parameters
        ----------
        population : list
            The list of individuals
        popSize: int
            Number of chrmosome in a population
        mutationProbability: float
            The probability of mutating an individual
        keep: int
            Number of best individuals to keep without mutating for the next generation
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list
        Returns
        -------
        N/A
        """
        ##for i in range(keep, popSize):
        for i in range(keep, self.SearchAgents_no):
            # Mutation
            offspringMutationProbability = random.uniform(0.0, 1.0)
            if offspringMutationProbability < mutationProbability:
                ##mutation(population[i], len(population[i]), lb, ub)
                mutation(self.Positions[i], len(self.Positions[i]), self.lb, self.ub)


    def elitism(population, scores, bestIndividual, bestScore):
        """
        This melitism operator of the population
        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        bestIndividual : list
            An individual of the previous generation having the best fitness value
        bestScore : float
            The best fitness value of the previous generation
        Returns
        -------
        N/A
        """

        # get the worst individual
        ##worstFitnessId = selectWorstIndividual(scores)
        worstFitnessId = selectWorstIndividual(self.fitness)

        # replace worst cromosome with best one from previous generation if its fitness is less than the other
        ##if scores[worstFitnessId] > bestScore:
        if self.fitness[worstFitnessId] > self.best_sol:
            ##population[worstFitnessId] = numpy.copy(bestIndividual)
            ##scores[worstFitnessId] = numpy.copy(bestScore)
            self.Positions[worstFitnessId] = numpy.copy(self.best_pos)
            self.fitness[worstFitnessId] = numpy.copy(self.best_sol )
            
            
    ##def selectWorstIndividual(scores):
    def selectWorstIndividual(self):
        
        
        """
        It is used to get the worst individual in a population based n the fitness value
        Parameters
        ----------
        scores : list
            The list of fitness values for each individual
        Returns
        -------
        int
           maxFitnessId: The individual id of the worst fitness value
        """

        maxFitnessId = numpy.where(scores == numpy.max(self.fitness))
        maxFitnessId = maxFitnessId[0][0]
        return maxFitnessId


    def pairSelection(population, scores, popSize):
        """
        This is used to select one pair of parents using roulette Wheel Selection mechanism
        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        popSize: int
            Number of chrmosome in a population
        Returns
        -------
        list
            parent1: The first parent individual of the pair
        list
            parent2: The second parent individual of the pair
        """
        ##parent1Id = rouletteWheelSelectionId(scores, popSize)
        ##parent1 = population[parent1Id].copy()
        parent1Id = rouletteWheelSelectionId(self.fitness, self.SearchAgents_no)
        parent1 = self.Positions[parent1Id].copy()

        ##parent2Id = rouletteWheelSelectionId(scores, popSize)
        ##parent2 = population[parent2Id].copy()
        parent2Id = rouletteWheelSelectionId(self.fitness, self.SearchAgents_no)
        parent2 = self.Positions[parent2Id].copy()

        return parent1, parent2


    def rouletteWheelSelectionId(scores, popSize):
        """
        A roulette Wheel Selection mechanism for selecting an individual
        
        Parameters
        ----------
        scores : list
            The list of fitness values for each individual
        popSize: int
            Number of chrmosome in a population
        Returns
        -------
        id
            individualId: The id of the individual selected
        """

        ##reverse score because minimum value should have more chance of selection
        ##reverse = max(scores) + min(scores)
        reverse = max(self.fitness) + min(self.fitness)
        ##reverseScores = reverse - scores.copy()
        reverseScores = reverse - self.fitness.copy()
        sumScores = sum(reverseScores)
        pick = random.uniform(0, sumScores)
        current = 0
        ##for individualId in range(popSize):
        for individualId in range(self.SearchAgents_no):
            current += reverseScores[individualId]
            if current > pick:
                return individualId
            
    def crossover(individualLength, parent1, parent2):
        """
        The crossover operator of a two individuals
        Parameters
        ----------
        individualLength: int
            The maximum index of the crossover
        parent1 : list
            The first parent individual of the pair
        parent2 : list
            The second parent individual of the pair
        Returns
        -------
        list
            offspring1: The first updated parent individual of the pair
        list
            offspring2: The second updated parent individual of the pair
        """

        # The point at which crossover takes place between two parents.
        crossover_point = random.randint(0, individualLength - 1)
        # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
        offspring1 = numpy.concatenate(
            [parent1[0:crossover_point], parent2[crossover_point:]]
        )
        # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
        offspring2 = numpy.concatenate(
            [parent2[0:crossover_point], parent1[crossover_point:]]
        )

        return offspring1, offspring2


    def mutation(offspring, individualLength, lb, ub):
        """
        The mutation operator of a single individual
        Parameters
        ----------
        offspring : list
            A generated individual after the crossover
        individualLength: int
            The maximum index of the crossover
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list
        Returns
        -------
        N/A
        """
        mutationIndex = random.randint(0, individualLength - 1)
        mutationValue = random.uniform(self.lb[mutationIndex], self.ub[mutationIndex])
        offspring[mutationIndex] = mutationValue

    def clearDups(Population, lb, ub):

        """
        It removes individuals duplicates and replace them with random ones
        
        Parameters
        ----------
        objf : function
            The objective function selected
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list
            
        Returns
        -------
        list
            newPopulation: the updated list of individuals
        """
        ##newPopulation = numpy.unique(Population, axis=0)
        ##oldLen = len(Population)
        ##newLen = len(newPopulation)
        
        newPopulation = numpy.unique(self.Positions, axis=0)
        oldLen = len(self.Positions)
        newLen = len(newPopulation)
        
        if newLen < oldLen:
            nDuplicates = oldLen - newLen
            newPopulation = numpy.append(
                newPopulation,
                ##numpy.random.uniform(0, 1, (nDuplicates, len(Population[0])))
                numpy.random.uniform(0, 1, (nDuplicates, len(self.Positions[0])))       
                * (numpy.array(self.ub) - numpy.array(self.lb))
                + numpy.array(self.lb),
                axis=0,
            )

        return newPopulation


    def calculateCost(objf, population, popSize, lb, ub):

        """
        It calculates the fitness value of each individual in the population
        Parameters
        ----------
        objf : function
            The objective function selected
        population : list
            The list of individuals
        popSize: int
            Number of chrmosomes in a population
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list
        Returns
        -------
        list
            scores: fitness values of all individuals in the population
        """
        ##scores = numpy.full(popSize, numpy.inf)
        self.fitness = numpy.full(self.SearchAgents_no, numpy.inf)

        # Loop through individuals in population
        ##for i in range(0, popSize):
        for i in range(0, self.SearchAgents_no):
            # Return back the search agents that go beyond the boundaries of the search space
            ##population[i] = numpy.clip(population[i], self.lb, self.ub)
            self.Position[i] = numpy.clip(self.Position[i], self.lb, self.ub)

            # Calculate objective function for each search agent
            ##scores[i] = self.objf(population[i, :])
            self.fitness[i] = self.objf(self.Position[i, :])

        ##return scores
        return self.fitness


    def sortPopulation(population, scores):
        """
        This is used to sort the population according to the fitness values of the individuals
        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        Returns
        -------
        list
            population: The new sorted list of individuals
        list
            scores: The new sorted list of fitness values of the individuals
        """
        ##sortedIndices = scores.argsort()
        ##population = population[sortedIndices]
        ##scores = scores[sortedIndices]
        
        sortedIndices = self.fitness.argsort()
        self.Position = self.Position[sortedIndices]
        self.fitness = self.fitness[sortedIndices]
        
        
        ##return population, scores
        return self.Position, self.fitness
    
    def GA(self):

        """
        This is the main method which implements GA
        Parameters
        ----------
        objf : function
            The objective function selected
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list
        dim: int
            The dimension of the indivisual
        popSize: int
            Number of chrmosomes in a population
        iters: int
            Number of iterations / generations of GA
        Returns
        -------
        obj
            s: The solution obtained from running the algorithm
        """

        cp = 1  # crossover Probability
        mp = 0.01  # Mutation Probability
        keep = 2
        # elitism parameter: how many of the best individuals to keep from one generation to the next

        ##s = solution()

        ##if not isinstance(lb, list):
            ##lb = [lb] * dim
        ##if not isinstance(ub, list):
            ##ub = [ub] * dim

        bestIndividual = numpy.zeros(self.dim)
        ##self.best_pos 
        
        ##scores = numpy.random.uniform(0.0, 1.0, self.SearchAgents_no)
        self.fitness = numpy.random.uniform(0.0, 1.0, self.SearchAgents_no)
        
        ##bestScore = float("inf")

        ga = numpy.zeros((self.SearchAgents_no, self.dim))
        for i in range(self.dim):
            ga[:, i] = numpy.random.uniform(0, 1, self.SearchAgents_no) * (self.ub[i] - self.lb[i]) + self.lb[i]
        self.convergence_curve = numpy.zeros(self.Max_iter)

        print('GA is optimizing  "' + self.objf.__name__ + '"')

        ##timerStart = time.time()
        ##s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        for self.t in range(self.Max_iter):

            # crossover
            ##ga = crossoverPopulaton(ga, scores, self.SearchAgents_no, cp, keep)
            ga = crossoverPopulaton(ga, self.fitness, self.SearchAgents_no, cp, keep)

            # mutation
            mutatePopulaton(ga, self.SearchAgents_no, mp, keep, self.lb, self.ub)

            ga = clearDups(ga, self.lb, self.ub)

            ##scores = calculateCost(self.objf, ga, self.SearchAgents_no, self.lb, self.ub)
            self.fitness = calculateCost(self.objf, ga, self.SearchAgents_no, self.lb, self.ub)

            ##bestScore = min(scores)
            self.best_sol = min(self.fitness)

            # Sort from best to worst
            ##ga, scores = sortPopulation(ga, scores)
            ga, scores = sortPopulation(ga, self.fitness)

            ##convergence_curve[l] = bestScore
            self.convergence_curve[self.t] = self.best_sol

            if self.t % 1 == 0:
                print(
                    [
                       "At iteration "
                       + str(self.t + 1)
                       + " the best fitness is "
                       + str(self.best_sol)
                    ]
                )
                
            # increase iterations
            self.t = self.t + 1
            self.s.R_list.append(1)

            if self.begin == True:
                reward = (self.convergence_curve[self.t - 2] - self.convergence_curve[self.t-1])/(self.convergence_curve[self.t - 2] +self.eps ) ## Revised Reward 
                self.Q[7] = self.ratio * reward + (1-self.ratio) * self.Q[7] 
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (GA): ", self.Q)
                self.recent.append(reward)
                self.optimize()
      
            if np.mod(self.t, self.K) == 1:
                reward = (self.convergence_curve[self.t - self.K-1] - self.convergence_curve[self.t-1]) / (self.convergence_curve[self.t - self.K-1]+self.eps) ### Revised Reward 
                self.Q[7] = self.ratio * reward + (1-self.ratio) * self.Q[7]  
                self.reward_list.append(reward)
                reward_avg = np.mean(self.reward_list)
                self.reward_avg_list.append(reward_avg)
                print("Values (GA): ", self.Q)

                term = np.sqrt((np.log(self.t))/self.count)
                ucb = self.Q+self.c*term
                print("UCB: ", ucb)
                #ind = np.ranind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])
                ind = np.random.choice(6, 1, p=[self.soft(ucb, 0), self.soft(ucb,1), self.soft(ucb, 2),self.soft(ucb,3),self.soft(ucb,4),self.soft(ucb,5),self.soft(ucb,6)])

                self.count[ind] +=1
                if ind == 0:
                    self.HHO()
                elif ind == 2:
                    self.WOA()
                elif ind == 1:
                    self.DE()
                elif ind == 3:
                    self.PSO()
                elif ind == 4:
                    self.JAYA()
                elif ind == 5:
                    self.FFA()
                elif ind == 6:
                    self.BAT()
           
            #7
            
        timerEnd = time.time()
        self.s.stopiter = self.f
        self.s.executionTime = timerEnd - self.timerStart
        self.s.convergence = self.convergence_curve   
            

            
        
        
  
                    
    
    

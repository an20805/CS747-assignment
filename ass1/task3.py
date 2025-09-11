"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed

def kl_divergence(p, q):
    if p == 0:
        return math.log(1 / (1 - q))
    if p == 1:
        return math.log(1 / q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

# ------------------ Optimized KL-UCB Algorithm ------------------

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """
    ## You can define other functions also in the class if needed
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        self.u = np.zeros(num_arms)
        self.kl_ucb = np.zeros(num_arms)
        self.p_hat = np.zeros(num_arms)
        self.t = 0
        self.n = 10
        self.chosen_arm = -1
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        if self.t < self.num_arms:
            return self.t
        else:
            if (self.t - self.num_arms) % self.n == 0:
                self.chosen_arm = np.argmax(self.kl_ucb)
            return self.chosen_arm

        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        self.t += 1
        self.u[arm_index] += 1
        self.p_hat[arm_index] = ((self.u[arm_index] - 1) * self.p_hat[arm_index] + reward) / self.u[arm_index]
        
        if self.t >= self.num_arms and (self.t - self.num_arms) % self.n == 0 and self.t <= self.horizon - self.horizon / self.num_arms:
            for i in range(self.num_arms):
                low, high = self.p_hat[i], 1.0
                threshold = (math.log(self.t) + 0.3 * math.log(math.log(self.t))) / self.u[i]
                mid = (low + high)/2

                while high - low > 1e-6:
                    if kl_divergence(self.p_hat[i], mid) > threshold:
                        high = mid
                    else:
                        low = mid
                    mid = (low + high)/2
                self.kl_ucb[i] = mid

        #END EDITING HERE

# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE

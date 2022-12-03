import random
import math
import numpy as np

# Get neighbor of current solution (X) by changing one variable assignment (0 -> 1 or vice versa)
def get_solution_neighbor(X):
    index_to_flip = random.randint(0, len(X)-1)
    X[index_to_flip] = 1 - X[index_to_flip]
    return X

def is_solution_feasible(X, machine_usage, resources):
    return np.all(machine_usage@X <= resources)

def get_score(X, rewards):
    return np.dot(X, rewards)

def solve_singleplayer_sim_anneal(instance, init_temp, max_iter):
    # Simulated annealing algorithm adapted from K&W Chapter 8.3 (pg. 130) - Algorithm 8.4
    rewards, machine_usage, resources = instance

    iter_count = 0
    temp = init_temp
    X, score = np.zeros(len(rewards)), 0
    best_X, best_score = X, score

    while iter_count < max_iter:
        new_X = get_solution_neighbor(X.copy())
        # continue process only if the neighbor solution is feasible; otherwise continue with iterations
        if is_solution_feasible(new_X, machine_usage, resources):
            # get difference between old/new scores
            new_score = get_score(new_X, rewards)
            score_diff = new_score - score
            # update X, score if needed (better score or random assignment)
            if score_diff > 0 or random.random() < math.exp(score_diff / temp):
                X, score = new_X, new_score
            # update best score if needed
            if score > best_score:
                best_X, best_score = X, score
        # update iter_count, temp
        iter_count += 1
        temp = init_temp / (iter_count + 1)

    return best_X

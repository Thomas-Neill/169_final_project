import random
import math 
import numpy as np

# Get neighbor of current solution (X) by changing one variable assignment (0 -> 1 or vice versa)
def get_01solution_neighbor(X):
    index_to_flip = random.randint(0, len(X)-1)
    X[index_to_flip] = 1 - X[index_to_flip]
    return X

def get_int_solution_neighbor(X):
    index_to_flip = random.randint(0, len(X)-1)
    X[index_to_flip] += random.choice([-1,1])
    return X

def is_solution_feasible(X, machine_usage, resources):
    return np.all(machine_usage@X <= resources)

def get_score(X, rewards):
    return np.dot(X, rewards)

def solve_sim_anneal(instance, init_temp, max_iter, get_neighbor,  X_start = None, return_stats=False):
    # Simulated annealing algorithm adapted from K&W Chapter 8.3 (pg. 130) - Algorithm 8.4
    rewards, machine_usage, resources = instance

    iter_count = 0
    temp = init_temp
    if X_start is None:
        X, score = np.zeros(len(rewards)), 0
    else:
        X,score = X_start, get_score(X_start,rewards)
    best_X, best_score = X, score
    old_best_score = score

    best_score_over_time = []

    while iter_count < max_iter:
        new_X = get_neighbor(X.copy())
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

        conv = abs(best_score - old_best_score)
        old_best_score = best_score
        best_score_over_time.append(best_score)
    
    if not return_stats:
        return best_X, best_score_over_time
    return best_score, conv, iter_count

def solve_singleplayer_sim_anneal(instance, init_temp=8000, max_iter=2000, return_stats = False):
    return solve_sim_anneal(instance, init_temp, max_iter, get_01solution_neighbor, return_stats=return_stats)

def solve_multiplayer_sim_anneal(instance, X_start, init_temp=8000, max_iter=2000, return_stats = False):
    return solve_sim_anneal(instance, init_temp, max_iter, get_int_solution_neighbor, X_start, return_stats)

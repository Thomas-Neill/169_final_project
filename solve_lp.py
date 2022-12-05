#this module contains various approximate integer LP solvers.
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt 
from genetic_algorithm import solve_singleplayer_lp_genetic
from simulated_annealing import solve_singleplayer_sim_anneal

def cvxpy_solve(instance):
    R, A, b = instance
    x = cp.Variable(len(R), integer = True)
    objective = cp.Maximize(x @ R)
    constraint = A@x <= b
    problem = cp.Problem(objective, [constraint])

    problem.solve(solver=cp.GLPK_MI)
    # According to wikipedia, the GNU linear programming kit
    # uses Gomory's mixed integer cuts method plus branch and bound.
    return x.value


if __name__ == '__main__':
    import instance_gen
    import singleplayer_lp
    import converters
    convs = instance_gen.gen_converters(4,3,2)
    ress = instance_gen.gen_resources(10)

    inst = singleplayer_lp.gen_instance(convs,ress)
    
    
    solution = cvxpy_solve(inst)
    if solution is not None:
        print(ress)
        print(solution)
        for i,x in enumerate(solution):
            if x:
                print("Used:", convs[i])
        print("Score:",np.dot(solution, inst[0]))
    
    # =================
    # genetic algorithm
    # =================
    
    print("\nGenetic Algorithm Solution\n")
    GA_MAX_ITERS = 100
    SHOW_PLOTS = True
    solution, statistics = solve_singleplayer_lp_genetic(
        inst, 
        max_population_size=100, 
        keep_top_k=20,
        max_iters=GA_MAX_ITERS,
        mutation_rate=1 / len(convs)
    )
    if solution is not None:
        # debug prints
        print(f"resources: {converters.resource_types}")
        print(f"resources: {inst[2]}")
        print(f"solution: {solution}")
        print(f"used_resources: {inst[1] @ solution}")
        
        
        for i,x in enumerate(solution):
            if x:
                print("Used:", convs[i])
        print("Score:",np.dot(solution, inst[0]))
        
        print("\nStatistics")
        for k,v in statistics.items():
            if type(v) is not list:
                print(f"  {k}: {v}")
                
        if SHOW_PLOTS:
            plt.subplot(2, 2, 1)
            # plot convergence over time
            plt.plot([i for i in range(GA_MAX_ITERS)], statistics["mean_convergence_over_time"])
            plt.title("mean convergence over time")
            
            # plot mean score over time
            plt.subplot(2, 2, 2)
            plt.plot([i for i in range(GA_MAX_ITERS)], statistics["mean_score_over_time"])
            plt.title("mean score over time")
            
             # plot best score over time
            plt.subplot(2, 2, 3)
            plt.plot([i for i in range(GA_MAX_ITERS)], statistics["best_score_over_time"])
            plt.title("best score over time")
            
            plt.show()
            
            
    else:
        print("found no solution...")
    
    # =================

    # =================
    # simulated annealing
    # =================
    
    solution_SA, best_score_over_time_SA = solve_singleplayer_sim_anneal(inst, init_temp=8000, max_iter=2000, return_stats=False)

    print("\nSimulated Annealing Solution\n")
    for i,x in enumerate(solution_SA):
        if x:
            print("Used:", convs[i])
    print("Score:",np.dot(solution_SA, inst[0]))

    if SHOW_PLOTS:
        plt.plot(list(range(len(best_score_over_time_SA))), best_score_over_time_SA)
        plt.title("best score over time")
        plt.show()
        
    # =================

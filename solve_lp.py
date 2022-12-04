#this module contains various approximate integer LP solvers.
import numpy as np
import cvxpy as cp
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
    # uses Gomory's mixed integer cuts plus branch and bound.

    return x.value


if __name__ == '__main__':
    import instance_gen
    import singleplayer_lp
    import converters
    convs = instance_gen.gen_converters(7,7,7)
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
    solution, statistics = solve_singleplayer_lp_genetic(
        inst, 
        max_population_size=100, 
        keep_top_k=20,
        max_iters=1000,
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
            print(f"  {k}: {v}")
    else:
        print("found no solution...")
    
    # =================

    # =================
    # simulated annealing
    # =================
    
    solution_sim_anneal = solve_singleplayer_sim_anneal(inst, init_temp=8000, max_iter=2000)

    print("\nSimulated Annealing Solution\n")
    for i,x in enumerate(solution_sim_anneal):
        if x:
            print("Used:", convs[i])
    print("Score:",np.dot(solution_sim_anneal, inst[0]))
        
    # =================
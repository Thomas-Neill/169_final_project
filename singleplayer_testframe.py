import time
import instance_gen, singleplayer_lp
from genetic_algorithm import *
from simulated_annealing import *
from solve_lp import *
import statistics
N_TESTS = 100

def genetic(inst):
    solution, statistics = solve_singleplayer_lp_genetic(
            inst, 
            max_population_size=100, 
            keep_top_k=20,
            max_iters=100,
            mutation_rate=1 / len(inst[0])
        )
    return solution @ inst[0], statistics["convergence"], statistics["iterations"]

def simulated(inst):
    return solve_singleplayer_sim_anneal(inst, init_temp=8000, max_iter=2000, return_stats=True)

def gnulptk(inst):
    return cvxpy_solve(inst)@inst[0], -1, -1

solvers = [("genetic", genetic),("simulated", simulated), ('cvxpy', gnulptk)]

results = {solver: [] for solver, solverfn in solvers}

for t in range(N_TESTS):
    if t % 10 == 0:
        print(t)
    convs = instance_gen.gen_converters(4,3,2)
    ress = instance_gen.gen_resources(10)

    inst = singleplayer_lp.gen_instance(convs,ress)
    for solver,solverframe in solvers:
        t0 = time.time()
        score, convergence, iterations = solverframe(inst)
        t1 = time.time()
        results[solver].append((score, convergence, t1-t0, iterations))
    #print(results)

for solver, _ in solvers:
    print(solver)
    scores = [i[0] for i in results[solver]]
    convs = [i[1] for i in results[solver]]
    dts = [i[2] for i in results[solver]]
    iters = [i[3] for i in results[solver]]
    for i in ["scores", "convs", "dts", "iters"]:
        print(i,statistics.mean(eval(i)),statistics.stdev(eval(i)))
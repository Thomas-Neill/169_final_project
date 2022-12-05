import time
import instance_gen, multiplayer_lp
from genetic_algorithm import *
from simulated_annealing import *
from solve_lp import *
import statistics
N_TESTS = 10
N_PLAYERS = 2

def genetic(inst):
    solution, statistics = solve_singleplayer_lp_genetic(
            inst, 
            max_population_size=100, 
            keep_top_k=20,
            max_iters=100,
            mutation_rate=1 / len(inst[0])
        )
    return solution @ inst[0], statistics["convergence"], statistics["iterations"]

def simulated(inst0, inst1, inst20, X0):
    return solve_multiplayer_sim_anneal(inst20, X0, init_temp=8000, max_iter=2000, return_stats=True)

def gnulptk(inst0, inst1, inst10, _ignore):
    if len(inst0[0]) <= 40:
        return cvxpy_solve(inst1)@inst0[0], -1, -1
    return cvxpy_solve(inst0)@inst0[0], -1, -1

solvers = [("simulated", simulated), ('cvxpy', gnulptk)]

results = {solver: [] for solver, solverfn in solvers}

for t in range(N_TESTS):
    if t % 1 == 0:
        print(t)
    players = [(instance_gen.gen_converters(4,3,2),instance_gen.gen_resources(10)) for p in range(N_PLAYERS)]

    inst0 = multiplayer_lp.gen_instance(players)[1]
    X0 = cvxpy_solve(inst0)
    base_scores = inst0[0] @ X0

    inst1 = multiplayer_lp.gen_instance(players, 1)[1]
    inst20 = multiplayer_lp.gen_instance(players, 20)[1]

    for solver,solverframe in solvers:
        print(solver)
        t0 = time.time()
        score, convergence, iterations = solverframe(inst0, inst1, inst20, X0)
        t1 = time.time()
        results[solver].append((score - base_scores, convergence, t1-t0, iterations))
    #print(results)

for solver, _ in solvers:
    print(solver)
    scores = [i[0] for i in results[solver]]
    convs = [i[1] for i in results[solver]]
    dts = [i[2] for i in results[solver]]
    iters = [i[3] for i in results[solver]]
    for i in ["scores", "convs", "dts", "iters"]:
        print(i,statistics.mean(eval(i)),statistics.stdev(eval(i)))
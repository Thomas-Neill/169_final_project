import time
import instance_gen, multiplayer_lp
from genetic_algorithm import *
from simulated_annealing import *
from solve_lp import *
import statistics
N_TESTS = 100
N_PLAYERS = 5

def genetic(inst0, inst2, inst20, X0):
    solution, statistics = solve_multiplayer_lp_genetic(
        inst20,
        max_population_size=100,
        keep_top_k=4,
        max_iters=100,
        mutation_rate=0.01,
        starting_solution= X0
    )
    if solution is None:
        return X0 @ inst0[0], statistics["convergence"], statistics["iterations"]
    return solution @ inst0[0], statistics["convergence"], statistics["iterations"]

def simulated(inst0, inst2, inst20, X0):
    return solve_multiplayer_sim_anneal(inst20, X0, init_temp=16000, max_iter=4000, return_stats=True)

def gnulptk(inst0, inst2, inst20, _ignore):
    if len(inst0[0]) <= 200:
        return cvxpy_solve(inst2)@inst0[0], -1, -1
    # it bricks if there are too many variables - just do a no-trade solve instead.
    return cvxpy_solve(inst0)@inst0[0], -1, -1

solvers = [("simulated", simulated), ('cvxpy', gnulptk), ('genetic', genetic)]

results = {solver: [] for solver, solverfn in solvers}

for t in range(N_TESTS):
    if t % 1 == 0:
        print(t)
    players = [(instance_gen.gen_converters(4,3,2),instance_gen.gen_resources(10)) for p in range(N_PLAYERS)]

    inst0 = multiplayer_lp.gen_instance(players)[1]
    X0 = cvxpy_solve(inst0)
    base_scores = inst0[0] @ X0

    inst2 = multiplayer_lp.gen_instance(players, 1)[1]
    inst20 = multiplayer_lp.gen_instance(players, 20)[1]

    for solver,solverframe in solvers:
        print(solver)
        t0 = time.time()
        score, convergence, iterations = solverframe(inst0, inst2, inst20, X0)
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
import time
N_TESTS = 10


solvers = []

results = {solver: [] for solver, solverfn in solvers}

for test_inst in N_TESTS:
    for solver,solverframe in solvers:
        t0 = time.time()
        score, convergence, iterations = solverframe(test_inst)
        t1 = time.time()
        results[solver].append((score, convergence, t1-t0, iterations))

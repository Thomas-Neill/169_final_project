import numpy as np
import converters
import solve_lp
import singleplayer_lp
from simulated_annealing import * 
import genetic_algorithm

def singleplayer_opt(player):
    inst = singleplayer_lp.gen_instance(*player)
    soln = solve_lp.cvxpy_solve(inst)
    return np.dot(soln, inst[0])

# list of tuples: (Convs_p1, Resources_p1)
# output: variable labels:
def gen_instance(players, max_trade = 0):
    # variables:
    # u_{p,m}: usage by player p of machine m
    # t_{p1, p2, r}: trade from player p1 to player p2 of resource r
    # t_{p1, p2, m}: trade from player p1 to player p2 of machine m

    # objective:
    # \sum_{p,m} u_{p,m} * v_m
    # constraints:
    # (1): \sum_m u{p,m} * i_{r,m} - \sum_{p2} t_{p2, p1, r} <= s_{p1, r}
    #     (no player overspends their resources)
    # (2): u_{p,m} - \sum_{p2} t_{p2, p1, m} <= s_{p1, m}
    #     (no player overuses their machines)
    # (3): \sum_m u_{p,m}*v_m >= indv_p (best individual score for that player)
    #     (each player gains something from having traded - TODO strengthen?)
    # (4): u_{p,m} >= 0
    #     (players use machines a positive number of times.)

    Np = len(players)
    machines = sorted(set(sum([i[0] for i in players],[])))

    variables = [(P, m) for m in machines for P in range(Np)] + \
        [(P1, P2, r) for r in converters.resource_types for P1 in range(Np) for P2 in range(P1+1,Np)] + \
        [(P1, P2, m) for m in machines for P1 in range(Np) for P2 in range(P1+1,Np)]

    objective = [v[1].output if len(v) == 2 else 0 for v in variables]

    constraint1 = [
        [v[1].inputs.count(r) if len(v) == 2 and v[0] == p else -1 if len(v) == 3 and v[1] == p and v[2] == r else 1 if len(v) == 3 and v[0] == p and v[2] == r else 0 for v in variables] 
        for r in converters.resource_types for p in range(Np)]
    
    limit1 = [players[p][1][r] for r in converters.resource_types for p in range(Np)]

    constraint2 = [[1 if v[0] == p and v[1] == m else -1 if len(v) == 3 and v[1] == p and v[2] == m else 1 if len(v) == 3 and v[0] == p and v[2] == m else 0 for v in variables] for m in machines for p in range(Np)]

    limit2 = [int(m in players[p][0]) for m in machines for p in range(Np)]

    constraint3 = [[-v[1].output if len(v) == 2 and v[0] == p else 0 for v in variables] for p in range(Np)]

    limit3 = [-singleplayer_opt(p) for p in players]

    constraint4 = [[-1 if v == v0 else 0 for v in variables] for v0 in variables if len(v0) == 2]

    limit4 = [0 for v0 in variables if len(v0) == 2]

    # constraint 5: trade is bounded: no trading in infinite loops!
    # this lets the solver terminate for N = 3, but needless to say, players can 
    # trade more than 1 resource!
    # todo fix somehow?

    constraint5 = [[-1 if v == v0 else 0 for v in variables] for v0 in variables if len(v0) == 3]

    limit5 = [max_trade for v0 in variables if len(v0) == 3]

    constraint6 = [[1 if v == v0 else 0 for v in variables] for v0 in variables if len(v0) == 3]

    limit6 = [1 for v0 in variables if len(v0) == 3]

    return (variables, (np.array(objective), 
            np.matrix(constraint1 + constraint2 + constraint3 + constraint4 + constraint5 + constraint6), 
            np.array(limit1 + limit2 + limit3 + limit4 + limit5 + limit6)))

if __name__ == '__main__':
    import instance_gen
    Np = 2
    players = [
        (sorted(set(instance_gen.gen_converters(4,4,4))),
         instance_gen.gen_resources(10)) for i in range(Np)]
    (vars,inst) = gen_instance(players)

    soln = solve_lp.cvxpy_solve(inst)

    print(soln)
    scores0 = []
    for i,p in enumerate(players):
        p1s = solve_lp.cvxpy_solve(singleplayer_lp.gen_instance(*p))
        '''for i,x in enumerate(p1s):
            if x:
                print("Used:", convs[i],x)'''
        scores0.append(singleplayer_opt(p))

    (vars, inst2) = gen_instance(players, 100)

    print("\nSimulated Annealing\n")
    soln2 = solve_multiplayer_sim_anneal(inst2, soln, 8000, 2000)

    scores = [0] * Np
    for x,v in zip(soln2,vars):
        if len(v) == 2 and x > 0:
            #print(v, x)
            scores[v[0]] += v[1].output*x
        #if len(v) == 3 and x > 0:
        #    print("!!!")
    print(scores0, sum(scores0))
    print(scores, sum(scores))
    
    print("\nGenetic Algorithm\n")
    soln3, _ = genetic_algorithm.solve_multiplayer_lp_genetic(
        inst2,
        max_population_size=300,
        keep_top_k=20,
        max_iters=400,
        mutation_rate=0.05,
        starting_solution=soln
    )
    
    if soln3 is not None:
        scores = [0] * Np
        for x,v in zip(soln3,vars):
            if len(v) == 2 and x > 0:
                #print(v, x)
                scores[v[0]] += v[1].output*x
            #if len(v) == 3 and x > 0:
            #    print("!!!")
        print(scores0, sum(scores0))
        print(scores, sum(scores))
    else:
        print("failed to find solution...")
    


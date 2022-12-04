import typing
import random
import numpy as np
import time

class Genome:
    __genes: typing.List[int]
    
    def __init__(self, genome_size: int, genes: typing.List[int] = []):
        self.__genes: typing.List[int] = genes if genes else [random.randint(0, 1) for _ in range(genome_size)]
        # check for programmer error
        if genes:
            assert len(genes) == genome_size, "genome size does not match passed in genes!"
        
    def get_usage_from_genes(self): #-> np.ndarray(float)
        return np.array([float(gene % 2) for gene in self.__genes])
        
    def compute_score(self, rewards, machine_usage, resources) -> float:
        """compute a final score for this genome and reward."""
        # calculate machine usage from genome: x in { 0.0, 1.0 }
        usage: np.ndarray[float] = self.get_usage_from_genes()
        # compute initial score
        score: int = np.dot(usage, rewards)
        # check constraint violation
        resource_usage: np.array = np.dot(machine_usage, usage)
        for i in range(resources.shape[0]):
            used, max = resource_usage[0, i], resources[i]
            if used > max:
                #print(f"resource_usage={resource_usage[i]} > resources={resources[i]}")
                score = 0
                break
    
        return score

    def get_genes(self) -> typing.List[int]:
        return self.__genes

    def set_gene(self, index: int, new_gene: int) -> None:
        self.__genes[index] = new_gene
        
    def get_gene(self, index: int) -> int:
        return self.__genes[index]

    def __len__(self) -> int:
        return len(self.__genes)

def selection(population, rewards, machine_usage, resources, keep_top_k: int) -> typing.List[typing.Tuple[Genome, Genome]]:
    """
    Select parents using truncated selection. Algorithm adapted from K&W Algorithm 9.6. 
    Returns tuples of parents that will be used in crossover to create children
    """
    # check for programmer error
    assert keep_top_k <= len(population), "you must keep more than the entire population"
    # score population
    fittest_pop: typing.List[Genome] = sorted(population, reverse=True, key=lambda x: x.compute_score(rewards, machine_usage, resources))[:keep_top_k]
    # compute random parents to be joined
    parent_indices: np.ndarray[int] = np.random.permutation(keep_top_k)
    selected: typing.List[typing.Tuple[Genome, Genome]] = [(fittest_pop[parent_indices[i]], fittest_pop[parent_indices[i + 1]]) for i in range(0, keep_top_k, 2)]
    return selected

def two_point_crossover(parent1: Genome, parent2: Genome) -> Genome:
    """Two point crossover. Algorithm adapted from K&W Algorithm 9.7"""
    genome_size = len(parent1)
    # compute two indices for the crossover
    point1, point2 = random.randint(0, genome_size), random.randint(0, genome_size)
    # swap if indices out of order
    if point1 > point2:
        point1, point2 = point2, point1
    # do crossover
    child_genes: typing.List[int] = parent1.get_genes()[:point1] + parent2.get_genes()[point1:point2] + parent1.get_genes()[point2:]
    return Genome(genome_size=genome_size, genes=child_genes)

def crossover(parents: typing.List[typing.Tuple[Genome, Genome]], population_size: int) -> typing.List[Genome]:
    """Creates a new population using crossover of selected parents."""
    children: typing.List[Genome] = []
    # create child from each of the parentss
    for p1, p2 in parents:
        children.append(two_point_crossover(parent1=p1, parent2=p2))
    
    # more children if population is too small
    while len(children) < population_size:
        random_parent_ind = random.randint(0, len(parents)-1)
        p1, p2 = parents[random_parent_ind]
        children.append(two_point_crossover(p1, p2))
    
    return children

def mutate(children: typing.List[Genome], mutation_rate: float) -> typing.List[Genome]:
    for child in children:
        for i in range(len(child)):
            if random.random() < mutation_rate:
                # bit flip on mutation
                child.set_gene(i, ~child.get_gene(i))
        
    return children

def __solver(instance, max_population_size: int, keep_top_k: int, max_iters: int, mutation_rate: float):
    """genetic algorithm solver for the single player lp. Algorithm adapted from K&W Algorithm 9.4."""
    
    # dictionary to hold statistics of optimization run
    statistics: typing.Dict[str, typing.Union[int, float]] = {}
    
    start_time = time.time()
    rewards, machine_usage, resources = instance
    # create an initial population
    population = [Genome(genome_size=machine_usage.shape[1]) for _ in range(max_population_size)]
    iters = 0
    prev_mean_score = 0
    convergence = 0
    convergence_over_iters: typing.List[float] = []
    mean_scores: typing.List[float] = []
    best_scores: typing.List[float] = []
    for i in range(1, max_iters+1):
        # select parents
        parents = selection(
            population, 
            rewards=rewards, 
            machine_usage=machine_usage, 
            resources=resources, 
            keep_top_k=keep_top_k
        )
        # crossover
        children = crossover(parents=parents, population_size=max_population_size)
        # mutate
        population = mutate(children, mutation_rate=mutation_rate)
        # track statistics
        iters = i
        
        scores = [x.compute_score(rewards, machine_usage, resources) for x in population]
        mean_score = sum(scores) / len(population)
        best_scores.append(max(scores))
        mean_scores.append(mean_score)
        convergence = abs(mean_score - prev_mean_score)
        convergence_over_iters.append(convergence)
        prev_mean_score = mean_score
    
    best: Genome = sorted(population, key=lambda x: x.compute_score(rewards, machine_usage, resources), reverse=True)[0]
    elapsed_time = time.time() - start_time
    
    statistics = { 
        "elapsed_time": elapsed_time, 
        "iterations": iters,
        "convergence": convergence,
        "mean_convergence_over_time": convergence_over_iters,
        "mean_score_over_time": mean_scores,
        "best_score_over_time": best_scores
    }
    
    return best.get_usage_from_genes() if best.compute_score(rewards, machine_usage, resources) > 0 else None, statistics
def solve_singleplayer_lp_genetic(instance, max_population_size: int, keep_top_k: int, max_iters: int, mutation_rate: float): #-> np.ndarray[float]
    return __solver(instance, max_population_size, keep_top_k, max_iters, mutation_rate)

def solve_multiplayer_lp_genetic(instance, max_population_size: int, keep_top_k: int, max_iters: int, mutation_rate: float): #-> np.ndarray[float]
    """genetic algorithm solver for the multiplayer lp. Algorithm adapted from K&W Algorithm 9.4."""
    assert False, "not implemented"
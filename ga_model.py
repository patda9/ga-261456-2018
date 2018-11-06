import numpy as np
np.random.seed(1)

# solution = [w1, w2, ..., wn]
class Individual(object): # create individual
    def __init__(self, w_shape):
        self.solution = np.random.randn(w_shape[0], w_shape[1])
        self.fitness = 0

def mate(mating_pool, mutation_rate):
    try:
        for i in range(0, len(mating_pool), 2):
            # print('before')
            # print(mating_pool[i].solution)
            # print(mating_pool[i+1].solution)
            cross_over(mating_pool[i], mating_pool[i+1])
            # print('after')
            # print(mating_pool[i].solution)
            # print(mating_pool[i+1].solution)
    except IndexError: # if there's odo number of mating pool member mutate the last one that has no pair
        # print('before mutation')
        # print(mating_pool[-1].solution)
        mutate(mutation_rate, mating_pool[-1], individual=True)
        # print('after mutation')
        # print(mating_pool[-1].solution)

def tournament_selection(d, n, population, population_size, y_hat): # n is desired number of offsprings
    p = 1 / population_size
    r = np.random.random()
    winner_pool = []

    index1 = np.random.randint(population_size)
    index2 = np.random.randint(population_size)
    while(index1 == index2):
        index2 = np.random.randint(population_size)
    
    calculate_fitness(d, population, y_hat)
    i = 0
    while(i < n):
        if(population[index1].fitness > population[index2].fitness):
            if(r < p):
                winner_pool.append(population[index2])
            else:
                winner_pool.append(population[index1])
        i += 1
    return winner_pool

def cross_over(i1, i2): # mating of individual1 and individual2
    crossing_index = np.random.randint(i1.solution.shape[0])
    
    while(crossing_index > i1.solution.shape[0] - 2): # prevent no crossing over
        crossing_index = np.random.randint(i1.solution.shape[0])
    # print('crossing site:', crossing_index)
    
    part1 = np.copy(i1.solution[crossing_index:])
    part2 = np.copy(i2.solution[crossing_index:])
    i1.solution[crossing_index:] = part2
    i2.solution[crossing_index:] = part1

def mutate(mutation_rate, population, individual=False):
    mutated_index = []
    if(individual == True): # use this block to mutate 1 individual
        mutate_index = np.random.randint(population.solution.shape[0])
        population.solution[mutate_index] = np.random.randn()
        mutated_index.append(mutate_index)
    else: # use this block to mutate random individual in population
        for i in range(population.shape[0]):
            if(np.random.random() < mutation_rate):
                mutate_index = np.random.randint(population.shape[0])
                population[i].solution[mutate_index] = np.random.randn()
                mutated_index.append(i)
    # print(mutated_index)

def sort_by_fitness(population):
    return np.sort(population.fitness, kind='mergesort')

def cross_entropy(y_hat, d, e=(1e-7)):
    y_hat = np.clip(y_hat, e, (1 - e))
    n = y_hat.shape[0]
    return -np.sum(d * np.log(y_hat + e)) / n

def calculate_fitness(d, population, y_hat):
    # return max: -cost(y_hat)
    for i in range(population.shape[0]):
        population[i].fitness = -cross_entropy(y_hat, d)

if(__name__ == '__main__'):
    iterations = 2048
    population = []

    size = 5
    mutation_rate = 1 / size

    for i in range(size):
        population.append(Individual((10, 1)))
    population = np.asarray(population)

    mate(population, mutation_rate)
    mutate(mutation_rate, population)

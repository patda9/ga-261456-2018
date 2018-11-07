import numpy as np
np.random.seed(2)

# solution = [w1, w2, ..., wn]
class Individual(object): # create individual
    def __init__(self, w_shape):
        self.solution = np.random.randn(w_shape[0], w_shape[1])
        self.fitness = 0

# train(x) -> cal_error(y_hat) -> fitness or cost(y_hat) -> update_with_ga(y_hat) -> train

def sigmoid(x, dx=False):
    if(dx == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def form_network(form, population):
    structure = form.split(',')
    weights = []
    
    i = 0
    n = 0
    for j in range(len(structure) - 1):
        structure[j] = int(structure[j])
        structure[j+1] = int(structure[j+1])
        n += structure[j] * structure[j+1]
        weights.append(population.solution[i:n].reshape(structure[j], structure[j+1]))
        i = n
    return weights

def forward_pass(d, form, population, x):
    y_hats = []
    for i in range(len(population)):
        weights = form_network(form, population[i])
        outputs = []

        z = x.dot(weights[0])
        a = sigmoid(z)
        outputs.append(a)
        for w in weights[1:]:
            input = z
            z = input.dot(w)
            a = sigmoid(z)
            outputs.append(a)
        y_hats.append(outputs[-1])
    return y_hats

def train(epochs, form, d, population, x): # epoch is generation
    j = 0
    while(j < epochs):
        print(population)
        # update fitness
        y_hats = forward_pass(d, form, population, x)
        for i in range(len(population)):
            population[i].fitness = calculate_fitness(d, population[i], y_hats[i])
            print(population[i].fitness)
        # sort population by fitness
        population = np.asarray(sort_by_fitness(population))

        ###
        # ga process goes below
        # ---------------------------------------------
        winner_pool, winner_index = tournament_selection(d, (len(population) / 2) + 1, population, len(population))
        print(winner_pool, winner_index)
        # mate(winner_pool, (1 / len(population)))
        ###

        optimal_solution_index = 0
        optimal_fitness = population[0].fitness
        optimal_solution = population[0]
        for i in range(1, len(population)):
            if(population[i].fitness > optimal_fitness):
                optimal_solution_index = i
                optimal_fitness = population[i].fitness
                optimal_solution = population[i]
        print('fitness:', optimal_fitness)
        print(('individual[' + str(optimal_solution_index) + ']:'), form_network(form, optimal_solution))
        j += 1

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
    except IndexError: # if there's odd number of mating pool member mutate the last one that has no pair
        # print('before mutation')
        # print(mating_pool[-1].solution)
        mutate(mutation_rate, mating_pool[-1], individual=True)
        # print('after mutation')
        # print(mating_pool[-1].solution)

def tournament_selection(d, n, population, population_size): # n is desired number of offsprings
    n = int(n)
    p = 1 / population_size
    winner_pool = []
    winner_indexes = []
    i = 0
    while(i < n):
        index1 = np.random.randint(population_size)
        index2 = np.random.randint(population_size)
        print((index1, index2))
        while(index1 == index2):
            index2 = np.random.randint(population_size)
        r = np.random.random()

        if(population[index1].fitness > population[index2].fitness):
            if(r < p):
                winner_pool.append(population[index2])
                winner_indexes.append(index2)
            else:
                winner_pool.append(population[index1])
                winner_indexes.append(index1)
        i += 1
    print('wi', winner_indexes)
    return winner_pool, winner_indexes

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
    return sorted(population, key=lambda x: x.fitness, reverse=True)

def cross_entropy(y_hat, d, e=(1e-7)):
    y_hat = np.clip(y_hat, e, (1 - e))
    n = y_hat.shape[0]
    return -np.sum(d * np.log(y_hat + e)) / n

def calculate_fitness(d, population, y_hat):
    # return max: -cost(y_hat)
    return -cross_entropy(y_hat, d)

if(__name__ == '__main__'):
    iterations = 2048
    population = []

    size = 10
    mutation_rate = 1 / size

    x = np.random.randn(10, 5)
    # print(x)
    d = np.random.randint(2, size=(10, 2))
    # print(d)

    for i in range(size):
        population.append(Individual((25, 1))) # 5 x 3 + 3 x 2
    population = np.asarray(population)

    mate(population, mutation_rate)
    mutate(mutation_rate, population)
    # forward_pass(d, '5, 3, 2, 2', population, x)
    train(3, '5, 3, 2, 2', d, population, x)

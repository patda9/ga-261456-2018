import numpy as np
np.random.seed(1)

# solution = [w1, w2, ..., wn]
class Population(object): # create individual
    def __init__(self, size, w_shape):
        self.solution = np.random.randn(w_shape[0], w_shape[1])
        self.fitness = 0

def cross_over(i1, i2): # mating of individual1 and individual2
    crossing_index = np.random.randint(i1.shape[0])
    part1 = np.copy(i1[crossing_index:])
    part2 = np.copy(i2[crossing_index:])
    i1[crossing_index:] = part2
    i2[crossing_index:] = part1

def mutate(mutation_rate, population):
    for i in range(population.shape[0]):
        if(np.random.random() < mutation_rate):
            mutate_index = np.random.randint(population.shape[0])
            population[i].solution[mutate_index] = np.random.randn()
        else:
            print('no mutation')

def sort_by_fitness(population):
    return np.sort(population.fitness, kind='mergesort')

size = 5
iterations = 2048
mutation_rate = 1 / size
population = np.asarray([Population(size, (10, 1))] * size)

print(population)
print(population[0].solution)
print(population[0].fitness)

i1 = np.random.randint(5, size=(5,1))
i2 = np.random.randint(5, high=99, size=(5,1))

cross_over(i1, i2)
mutate(mutation_rate, population)

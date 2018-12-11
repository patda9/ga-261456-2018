import numpy as np
# np.random.seed(2)

# solution = [w1, w2, ..., wn]
class Individual(object): # create individual
    def __init__(self, w_shape):
        self.solution = np.random.randn(w_shape[0], w_shape[1])
        self.fitness = -(np.inf)

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

def forward_pass(d, form, population, x, model=False):
    y_hats = []
    if(model == True):
        optimal_solution = population
        outputs = []
        weights = form_network(form, optimal_solution)

        z = x.dot(weights[0])
        a = sigmoid(z)
        outputs.append(a)
        for w in weights[1:]:
            input = z
            z = input.dot(w)
            a = sigmoid(z)
            outputs.append(a)
        y_hats.append(outputs[-1])
    else:
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
    optimal_solution_index = 0
    optimal_fitness = -(np.inf)
    optimal_solution = population[0]
    
    while(j < epochs):
        # update fitness
        # print(j)
        # for i in range(len(population)):
        #     print(population[i].solution)
        #     print(population[i].fitness)
        # print('before')
        # for i in range(len(population)):
            # print(population[i].solution)
            # print(population[i].fitness)

        y_hats = forward_pass(d, form, population, x)
        for i in range(len(population)):
            population[i].fitness = calculate_fitness(d, population[i], y_hats[i])
        # sort population by fitness
        population = np.asarray(sort_by_fitness(population))

        ###
        # ga process goes below
        # ---------------------------------------------
        winner_pool, winner_index = tournament_selection(d, epochs, ((len(population) / 2) + 1), population, len(population))
        # print('***', winner_pool[0].solution)
        # for i in range(len(winner_pool)):
            # print(winner_pool[i].solution)
        mate(winner_pool, 1 / len(population))
        # print()
        # for i in range(len(winner_pool)):
            # print(winner_pool[i].solution)
        elite_index = len(population) - int((len(population) / 2) + 1)
        k = 0
        for i in range(elite_index, len(population)):
            # print(population[i].solution)
            population[i].solution = winner_pool[k].solution
            k += 1
            # print(population[i].solution)
        # print(list(set(winner_pool)), list(set(winner_index)))
        # print(winner_pool[0].solution)
        # mate(winner_pool, (1 / len(population)))
        ###

        # print('after')
        # for i in range(len(population)):
            # print(population[i].solution)
            # print(population[i].fitness)
        
        for i in range(1, len(population)):
            if(population[i].fitness > optimal_fitness):
                optimal_solution_index = i
                optimal_fitness = population[i].fitness
                optimal_solution = population[i]
        if(j % 128 == 0):
            print('fitness:', population[0].fitness)
        # print(('individual[' + str(optimal_solution_index) + ']:'), form_network(form, optimal_solution))
        j += 1
    print(optimal_fitness)
    print(optimal_solution_index)
    print(optimal_solution.solution)
    return optimal_solution

def mate(mating_pool, mutation_rate):
    try:
        for i in range(0, len(mating_pool), 2):
            cross_over(mating_pool[i], mating_pool[i+1])

    except IndexError: # if there's odd number of mating pool member mutate the last one that has no pair
        mutate(mutation_rate, mating_pool[-1], individual=True)

def tournament_selection(d, iteration, n, population, population_size): # n is desired number of offsprings
    n = int(n)
    p = .5 * np.exp(-.5 * iteration)
    winner_pool = []
    winner_indexes = []
    i = 0
    while(i < n):
        index1 = np.random.randint(population_size)
        index2 = np.random.randint(population_size)
        while(index1 == index2):
            index2 = np.random.randint(population_size)
        r = np.random.random()

        if(population[index1].fitness > population[index2].fitness):
            if(r < p):
                winner_pool.append(population[index2])
                winner_indexes.append(index2)
                # print((index2, index1))
            else:
                winner_pool.append(population[index1])
                winner_indexes.append(index1)
                # print((index1, index2))
        else:
            winner_pool.append(population[index2])
            winner_indexes.append(index2)
            # print((index2, index1))
        i += 1
    # print('wi', winner_indexes)
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

def predict(y_hat):
    if(y_hat[0] >= .5):
        return 1.
    else:
        return 0.

def k_fold(x, d, ):
    # combine 
    # shuffle
    # folding 

    # fold_len = int(data.shape[0] / k)
    # input_folds = []
    # output_folds = []
    # for i in range(k):
    #     if(i >= k-1):
    #         input_folds += [input[i * fold_len:(i+1) * fold_len + (input.shape[0] % fold_len)]]
    #         output_folds += [d[i * fold_len:(i+1) * fold_len + (input.shape[0] % fold_len)]]
    #     input_folds += [input[i * fold_len:(i+1) * fold_len]]
    #     output_folds += [d[i * fold_len:(i+1) * fold_len]]

    # if(input.shape[0] % k > 0): # prevent empty array
    #     input_folds += [input[k * fold_len:input.shape[0]]]
    #     output_folds += [d[k * fold_len:d.shape[0]]]

    # for i in range(k):
    #     print('fold:', i)
    #     input_temp = input_folds.copy()
    #     output_temp = output_folds.copy()
    #     testing_set = input_temp[i]
    #     d_test = output_temp[i]
    #     del(input_temp[i])
    #     del(output_temp[i])
    #     training_set = np.concatenate(input_temp, axis=0)
    #     d_train = np.concatenate(output_temp, axis=0)
    #     layers = train(activations, form, training_set, d_train, learning_rate, epochs)
    #     fold_acc = test(layers, testing_set, d_test)
    #     sum_acc += fold_acc
    #     print('fold[' + str(i) + '] accuracy:', fold_acc, '%', '\n', '\n')

if(__name__ == '__main__'):
    iterations = 1000
    population = []

    size = 100
    mutation_rate = 1 / size

    wdbc_data = np.genfromtxt('wdbc-norm.csv', delimiter=',')
    x = np.copy(wdbc_data).T
    x = x[1:].T
    # print(x)
    d = (np.copy(wdbc_data)).T
    d = d[0].reshape(d[0].shape[0], 1)
    # print(d)

    hidden = [3, 1]

    for i in range(size):
        population.append(Individual((x.shape[1] * hidden[0] + hidden[0] * hidden[1], 1))) # 5 x 3 + 3 x 2
    population = np.asarray(population)

    form = (str(x.shape[1]) + ', ' + str(hidden[0]) + ', ' + str(hidden[1]))
    optimal_solution = train(iterations, fo b rm, d, population, x)

    test_x = x[np.random.randint(x.shape[0])]
    output_class = forward_pass(d, form, optimal_solution, test_x, model=True)
    print(predict(output_class))

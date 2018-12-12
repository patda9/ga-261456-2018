import numpy as np
# np.random.seed(2)

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

def train(epochs, form, d, population, x, iteration): # epoch is generation
    j = 0
    optimal_solution_index = 0
    optimal_fitness = 10000
    optimal_solution = population[0]
    
    file = open('training-log-fold' + str(iteration) + '.txt', 'w')
    while(j < epochs):
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
        for i in range(1, len(population)):
            if(population[i].fitness < optimal_fitness):
                print(population[i].fitness, optimal_fitness)
                optimal_solution_index = i
                optimal_fitness = population[i].fitness
                optimal_solution = population[i]
        print('*', optimal_solution.fitness)
        print(j)
        if(j % 256 == 0):
            print('fitness:', optimal_solution.fitness)
        # print(('individual[' + str(optimal_solution_index) + ']:'), form_network(form, optimal_solution))
        j += 1
    file.writelines('fitness:' + str(optimal_solution.fitness) + '\n')
    print(optimal_fitness)
    print(optimal_solution_index)
    print(optimal_solution.solution)
    file.close()
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
    return cross_entropy(y_hat, d)

def predict(y_hats):
    for i in range(len(y_hats)):
        for j in range(len(y_hats[i])):
            if(y_hats[i][j] >= .5):
                y_hats[i][j] = 1
            else:
                y_hats[i][j] = 0
    return y_hats

def accuracy_measure(d_test, predictions):
    compare = np.concatenate((d_test, predictions), axis=1)
    count = 0
    print(compare)
    for i in range(len(d_test)):
        for j in range(len(d_test[i])):
            if(d_test[i][j] == predictions[i][j]):
                count += 1
    return count / len(predictions)

def test(d_test, form, optimal_solution, x_test):
    y_hats = forward_pass(d_test, form, optimal_solution, x_test, model=True)[0]
    predictions = predict(y_hats)
    accuracy = accuracy_measure(d_test, predictions)
    return accuracy

def k_fold(x, d, k, epochs, form, population):
    pareto = []
    accuracies = []
    # combine 
    data = np.concatenate((x, d), axis=1)
    # shuffle
    np.random.shuffle(data)
    #separate
    x = np.hsplit(data, [x.shape[1]])[0]
    d = np.hsplit(data, [x.shape[1]])[1].reshape(d.shape)
    # folding 
    fold_len = int(data.shape[0] / k)
    x_folds = []
    d_folds = []
    for i in range(k):
        if(i >= k-1):
            x_folds += [x[i * fold_len:(i+1) * fold_len + (x.shape[0] % fold_len)]]
            d_folds += [d[i * fold_len:(i+1) * fold_len + (x.shape[0] % fold_len)]]
        x_folds += [x[i * fold_len:(i+1) * fold_len]]
        d_folds += [d[i * fold_len:(i+1) * fold_len]]

    if(x.shape[0] % k > 0): # prevent empty array
        x_folds += [x[k * fold_len:x.shape[0]]]
        d_folds += [d[k * fold_len:d.shape[0]]]
    
    print(len(x_folds[-1]))
    print(len(d_folds[-1]))

    for i in range(k):
        print('fold:', i)
        x_temp = x_folds.copy()
        d_temp = d_folds.copy()
        x_test = x_temp[i]
        d_test = d_temp[i]
        del(x_temp[i])
        del(d_temp[i])
        x_train = np.concatenate(x_temp, axis=0)
        d_train = np.concatenate(d_temp, axis=0)

        optimal_solution = train(epochs, form, d_train, population, x_train, i) # epoch is generation
        accuracy = test(d_test, form, optimal_solution, x_test)

        pareto.append(optimal_solution)
        accuracies.append(accuracy)
        file = open('./output-fold' + str(i) + '.txt', 'w')
        file.writelines('optimal solution:' + str(optimal_solution.solution) + '\n')
        file.writelines('fitness:' + str(optimal_solution.fitness) + '\n')
        file.writelines('fold accuracy:' + str(accuracy) + '\n')
        file.close()
    return pareto, accuracies

if(__name__ == '__main__'):
    iterations = 2048
    population = []

    size = 256
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
        population.append(Individual((x.shape[1] * hidden[0] + hidden[0] * hidden[1], 1))) # 30 x 3 + 3 x 1
    population = np.asarray(population)

    form = (str(x.shape[1]) + ', ' + str(hidden[0]) + ', ' + str(hidden[1]))
    pareto_front, accuracies = k_fold(x, d, 10, iterations, form, population)

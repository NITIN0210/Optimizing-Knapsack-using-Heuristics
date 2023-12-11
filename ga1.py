import random
import time
import re

# Parameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# Knapsack problem data
'''
weights = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
values = [25, 35, 55, 75, 95, 115, 135, 155, 175, 195]
capacity = 100'''


# Knapsack objective function
def knapsack_fitness(solution,weights,values,capacity):
    total_weight = 0
    total_value = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            total_weight += weights[i]
            total_value += values[i]
    if total_weight > capacity:
        return 0
    else:
        return total_value


# regular expression to match the header of each instance
header_pattern = re.compile(r'^f(\d+)_(l-d)_kp_(\d+)_(\d+)$')

instances = []

with open('d1.txt') as f:
    lines = f.readlines()
    i = 0
    j=0
    while i < len(lines):# or 
        # search for the header of the instance
        header_match = header_pattern.match(lines[i])
        if header_match:
            # extract the number of items and knapsack capacity from the header
            num_items = int(header_match.group(3))
            knapsack_size = int(header_match.group(4))
            # read the weights and values arrays
            weights = []
            values = []
            for j in range(i+1, i+num_items+1):
                value, weight = map(int, lines[j].split())
                weights.append(weight)
                values.append(value)
            # store the instance information in a dictionary
            instance = {
                'num_items': num_items,
                'knapsack_size': knapsack_size,
                'weights': weights,
                'values': values
            }
            instances.append(instance)
            # move to the next instance
            i += num_items+2
        else:
            # move to the next line
            i += 1
c=0
for instance in instances:
    weights = instance["weights"]
    values = instance["values"]
    capacity = instance["knapsack_size"]
    print(weights,values,capacity)
    # Genetic Algorithm
    population = []
    for i in range(POPULATION_SIZE):
        solution = [random.randint(0, 1) for j in range(len(weights))]
        population.append(solution)

    start_time = time.time()
    best_fitness = 0
    best_solution = []
    for generation in range(NUM_GENERATIONS):
        # Selection
        parents = []
        for i in range(POPULATION_SIZE):
            tournament = random.sample(population, 2)
            if knapsack_fitness(tournament[0],weights,values,capacity) > knapsack_fitness(tournament[1],weights,values,capacity):
                parents.append(tournament[0])
            else:
                parents.append(tournament[1])

        # Crossover
        offspring = []
        for i in range(POPULATION_SIZE):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            if random.random() < CROSSOVER_RATE:
                crossover_point = random.randint(0, len(weights) - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child = parent1
            offspring.append(child)

        # Mutation
        for i in range(POPULATION_SIZE):
            for j in range(len(weights)):
                if random.random() < MUTATION_RATE:
                    offspring[i][j] = 1 - offspring[i][j]

        # Evaluation
        for i in range(POPULATION_SIZE):
            fitness = knapsack_fitness(offspring[i],weights,values,capacity)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = offspring[i]

        # Replacement
        population = offspring

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Output
    print('best solution',best_solution)
    print("Iteration: ", NUM_GENERATIONS)
    print("Time: ", elapsed_time)
    print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
    print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
    print("Objective: ", best_fitness)
    c=c+1
    
    print(c)

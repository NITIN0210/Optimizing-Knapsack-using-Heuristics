import random
import time
import re
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


# Parameters
POPULATION_SIZE = 50
NUM_ITERATIONS = 100
NUM_REF_SETS = 5
REF_SET_SIZE = 10

# Knapsack problem data

# regular expression to match the header of each instance
#header_pattern = re.compile(r'^f(\d+)-(hd|ld)-kp-(\d+)-(\d+)$')
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
                weight, value = map(int, lines[j].split())
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

# Local Search
c=0
for instance in instances:
    weights = instance["weights"]
    values = instance["values"]
    capacity = instance["knapsack_size"]
    population = []
    for i in range(POPULATION_SIZE):
        solution = [random.randint(0, 1) for j in range(len(weights))]
        population.append(solution)

    # Scatter Search
    start_time = time.time()
    for iteration in range(NUM_ITERATIONS):
        # Generate reference sets
        ref_sets = []
        for i in range(NUM_REF_SETS):
            candidates = []
            for j in range(REF_SET_SIZE):
                solution = [random.randint(0, 1) for k in range(len(weights))]
                candidates.append(solution)
            ref_sets.append(candidates)

        # Merge reference sets and sort by fitness
        candidates = [candidate for ref_set in ref_sets for candidate in ref_set]
        candidates_fitness = [knapsack_fitness(candidate,weights,values,capacity) for candidate in candidates]
        sorted_candidates = [x for _, x in sorted(zip(candidates_fitness, candidates), reverse=True)]

        # Select new population
        population = sorted_candidates[:POPULATION_SIZE]

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Select best solution
    best_solution = []
    best_fitness = 0
    for i in range(1, len(population)):
        fitness = knapsack_fitness(population[i],weights,values,capacity)
        if fitness > best_fitness:
            best_solution = population[i]
            best_fitness = fitness

    # Output
    print("Iteration: ", NUM_ITERATIONS)
    print("Time: ", elapsed_time)
    print('capacity:',capacity)
    print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
    print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
    print("Objective2 profits: ", sum([values[i] for i in range(len(best_solution)) if best_solution[i]==1]))

    print("Objective: ", best_fitness)

    c=c+1
    print(c)

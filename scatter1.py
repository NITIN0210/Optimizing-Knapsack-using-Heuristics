import random
import time

# Parameters
POPULATION_SIZE = 50
NUM_ITERATIONS = 100
NUM_REF_SETS = 5
REF_SET_SIZE = 10

# Knapsack problem data
weights = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
values = [25, 35, 55, 75, 95, 115, 135, 155, 175, 195]
capacity = 100

# Knapsack objective function
def knapsack_fitness(solution):
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

# Generate initial population
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
    candidates_fitness = [knapsack_fitness(candidate) for candidate in candidates]
    sorted_candidates = [x for _, x in sorted(zip(candidates_fitness, candidates), reverse=True)]

    # Select new population
    population = sorted_candidates[:POPULATION_SIZE]

end_time = time.time()
elapsed_time = end_time - start_time

# Select best solution
best_solution = population[0]
best_fitness = knapsack_fitness(best_solution)
for i in range(1, len(population)):
    fitness = knapsack_fitness(population[i])
    if fitness > best_fitness:
        best_solution = population[i]
        best_fitness = fitness

# Output
print("Iteration: ", NUM_ITERATIONS)
print("Time: ", elapsed_time)
print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
print("Objective: ", best_fitness)

import random
import time

# Parameters
TABU_TENURE = 10
NUM_ITERATIONS = 1000

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

# Tabu Search
solution = [random.randint(0, 1) for j in range(len(weights))]
tabu_list = [set() for j in range(len(weights))]
best_fitness = knapsack_fitness(solution)
best_solution = solution

start_time = time.time()
for iteration in range(NUM_ITERATIONS):
    # Evaluate neighbors
    neighbors = []
    for i in range(len(weights)):
        neighbor = solution[:]
        neighbor[i] = 1 - neighbor[i]
        if tuple(neighbor) not in tabu_list[i]:
            neighbors.append(neighbor)
    neighbors_fitness = [knapsack_fitness(neighbor) for neighbor in neighbors]

    # Select best neighbor
    best_neighbor = neighbors[0]
    best_neighbor_fitness = neighbors_fitness[0]
    for i in range(1, len(neighbors)):
        if neighbors_fitness[i] > best_neighbor_fitness:
            best_neighbor = neighbors[i]
            best_neighbor_fitness = neighbors_fitness[i]

    # Update tabu list
    for i in range(len(weights)):
        if solution[i] != best_neighbor[i]:
            tabu_list[i].add(tuple(solution))
            if len(tabu_list[i]) > TABU_TENURE:
                tabu_list[i].remove(sorted(tabu_list[i])[0])
    solution = best_neighbor
    fitness = knapsack_fitness(solution)
    if fitness > best_fitness:
        best_fitness = fitness
        best_solution = solution

end_time = time.time()
elapsed_time = end_time - start_time

# Output
print("Iteration: ", NUM_ITERATIONS)
print("Time: ", elapsed_time)
print('capacity:',capacity)
print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
print("Objective2 profits: ", sum([values[i] for i in range(len(best_solution)) if best_solution[i]==1]))

print("Objective: ", best_fitness)

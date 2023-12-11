import re
import random 
import time
import numpy as np
import matplotlib.pyplot as plt


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

def file_read():
    header_pattern = re.compile(r'^knapPI_(\d+)_(\d+)_(\d+)_(\d+)$')

    instances = []
    fnames = ['data1_1.txt']
    
    for fname in fnames:
        with open(fname) as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                header_match = header_pattern.match(lines[i])
                if header_match:
                    num_items = int(header_match.group(2))
                    knapsack_size = int(header_match.group(3))

                    weights = []
                    values = []
                    for j in range(i+1, i+num_items+1):
                        value, weight = map(int, lines[j].split())
                        weights.append(weight)
                        values.append(value)
                    
                    instance = {
                        'num_items': num_items,
                        'knapsack_size': knapsack_size,
                        'weights': weights,
                        'values': values
                    }
                    instances.append(instance)
                    i=i+num_items+1
                else:
                    i += 1
    return instances

instances = file_read()

def GA(instances):
    
    POPULATION_SIZE = 1000
    NUM_GENERATIONS = 10000
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    ga=[]
    c=0
    for instance in instances:
        weights = instance["weights"]
        values = instance["values"]
        capacity = instance["knapsack_size"]
        population = []
        for i in range(POPULATION_SIZE):
            solution = [random.randint(0, 1) for j in range(len(weights))]
            population.append(solution)

        start_time = time.time()
        best_fitness = 0
        best_solution = []
        for generation in range(NUM_GENERATIONS):
            parents = []
            for i in range(POPULATION_SIZE):
                tournament = random.sample(population, 2)
                if knapsack_fitness(tournament[0],weights,values,capacity) > knapsack_fitness(tournament[1],weights,values,capacity):
                    parents.append(tournament[0])
                else:
                    parents.append(tournament[1])

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

            for i in range(POPULATION_SIZE):
                for j in range(len(weights)):
                    if random.random() < MUTATION_RATE:
                        offspring[i][j] = 1 - offspring[i][j]

            for i in range(POPULATION_SIZE):
                fitness = knapsack_fitness(offspring[i],weights,values,capacity)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = offspring[i]

            population = offspring

        end_time = time.time()
        elapsed_time = end_time - start_time

        instanceoutput={
            'time':elapsed_time,
            'capacity':capacity,
            'items':[i+1 for i in range(len(best_solution)) if best_solution[i]==1],
            'weight':sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]),
            'objective':best_fitness

        }
        ga.append(instanceoutput)
    return ga

def scatter(instances):

    POPULATION_SIZE = 500
    NUM_ITERATIONS = 5000
    NUM_REF_SETS = 5
    REF_SET_SIZE = 10
    s=[]
    c=0
    for instance in instances:
        weights = instance["weights"]
        values = instance["values"]
        capacity = instance["knapsack_size"]
        population = []
        for i in range(POPULATION_SIZE):
            solution = [random.randint(0, 1) for j in range(len(weights))]
            population.append(solution)

        start_time = time.time()
        for iteration in range(NUM_ITERATIONS):
            ref_sets = []
            for i in range(NUM_REF_SETS):
                candidates = []
                for j in range(REF_SET_SIZE):
                    solution = [random.randint(0, 1) for k in range(len(weights))]
                    candidates.append(solution)
                ref_sets.append(candidates)

            candidates = [candidate for ref_set in ref_sets for candidate in ref_set]
            candidates_fitness = [knapsack_fitness(candidate,weights,values,capacity) for candidate in candidates]
            sorted_candidates = [x for _, x in sorted(zip(candidates_fitness, candidates), reverse=True)]

            population = sorted_candidates[:POPULATION_SIZE]

        end_time = time.time()
        elapsed_time = end_time - start_time

        best_solution = []
        best_fitness = 0
        for i in range(1, len(population)):
            fitness = knapsack_fitness(population[i],weights,values,capacity)
            if fitness > best_fitness:
                best_solution = population[i]
                best_fitness = fitness

        instanceoutput={
            'time':elapsed_time,
            'capacity':capacity,
            'items':[i+1 for i in range(len(best_solution)) if best_solution[i]==1],
            'weight':sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]),
            'objective':best_fitness

        }
        s.append(instanceoutput)
    return s

def local(instances):

    NUM_ITERATIONS = 5000
    l=[]

    c=0
    for instance in instances:

        weights = instance["weights"]
        values = instance["values"]
        capacity = instance["knapsack_size"]
        
        solution = [random.randint(0, 1) for j in range(len(weights))]
        best_fitness = knapsack_fitness(solution,weights,values,capacity)
        best_solution = solution

        start_time = time.time()
        for iteration in range(NUM_ITERATIONS):
            neighbors = []
            for i in range(len(weights)):
                neighbor = solution[:]
                neighbor[i] = 1 - neighbor[i]
                neighbors.append(neighbor)
            neighbors_fitness = [knapsack_fitness(neighbor,weights,values,capacity) for neighbor in neighbors]

            best_neighbor = neighbors[0]
            best_neighbor_fitness = neighbors_fitness[0]
            for i in range(1, len(neighbors)):
                if neighbors_fitness[i] > best_neighbor_fitness:
                    best_neighbor = neighbors[i]
                    best_neighbor_fitness = neighbors_fitness[i]

            solution = best_neighbor
            fitness = knapsack_fitness(solution,weights,values,capacity)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution

        end_time = time.time()
        elapsed_time = end_time - start_time

        instanceoutput={
            'time':elapsed_time,
            'capacity':capacity,
            'items':[i+1 for i in range(len(best_solution)) if best_solution[i]==1],
            'weight':sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]),
            'objective':best_fitness

        }
        l.append(instanceoutput)
    return l

def tabu(instances):

    TABU_TENURE = 1000
    NUM_ITERATIONS = 10000
    t=[]

    c=0
    for instance in instances:
        weights = instance["weights"]
        values = instance["values"]
        capacity = instance["knapsack_size"]
        
        solution = [random.randint(0, 1) for j in range(len(weights))]
        tabu_list = [set() for j in range(len(weights))]
        best_fitness = knapsack_fitness(solution,weights,values,capacity)
        best_solution = solution

        start_time = time.time()
        for iteration in range(NUM_ITERATIONS):
            neighbors = []
            for i in range(len(weights)):
                neighbor = solution[:]
                neighbor[i] = 1 - neighbor[i]
                if tuple(neighbor) not in tabu_list[i]:
                    neighbors.append(neighbor)
            neighbors_fitness = [knapsack_fitness(neighbor,weights,values,capacity) for neighbor in neighbors]

            best_neighbor = neighbors[0]
            best_neighbor_fitness = neighbors_fitness[0]
            for i in range(1, len(neighbors)):
                if neighbors_fitness[i] > best_neighbor_fitness:
                    best_neighbor = neighbors[i]
                    best_neighbor_fitness = neighbors_fitness[i]

            for i in range(len(weights)):
                if solution[i] != best_neighbor[i]:
                    tabu_list[i].add(tuple(solution))
                    if len(tabu_list[i]) > TABU_TENURE:
                        tabu_list[i].remove(sorted(tabu_list[i])[0])
            solution = best_neighbor
            fitness = knapsack_fitness(solution,weights,values,capacity)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution

        end_time = time.time()
        elapsed_time = end_time - start_time

        instanceoutput={
            'time':elapsed_time,
            'capacity':capacity,
            'items':[i+1 for i in range(len(best_solution)) if best_solution[i]==1],
            'weight':sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]),
            'objective':best_fitness
        }
        t.append(instanceoutput)
    return t

ga = GA(instances)
t = tabu(instances)
s = scatter(instances)
l = local(instances)

print('=========================')
for i in range(len(ga)):
    print('',ga[i],'\n',t[i],'\n',s[i],'\n',l[i])
    print('---------')

algos = ["Genetic", "Tabu", "Local", "Scatter"]

# 1. Time Comparison
time_data = [ga, t, l, s]
times = [np.mean([d['time'] for d in data]) for data in time_data]
plt.bar(algos, times)
plt.title("Average Time Taken")
plt.ylabel("Time (s)")
plt.show()

# 2. Objective Value Comparison
fig, ax = plt.subplots()
for i, data in enumerate([ga, t, l, s]):
    obj_values = [d['objective'] for d in data]
    ax.scatter(np.arange(1, len(instances)+1), obj_values, label=algos[i])
ax.legend()
ax.set_xlabel("Instance")
ax.set_ylabel("Objective Value")
ax.set_title("Objective Value Comparison")
plt.show()

# 3. Solution Quality
data = [ [d['objective'] for d in ga],
         [d['objective'] for d in t],
         [d['objective'] for d in l],
         [d['objective'] for d in s] ]
plt.boxplot(data, labels=algos)
plt.ylabel("Objective Value")
plt.title("Solution Quality Comparison")
plt.show()

# 4. Convergence Plot
fig, ax = plt.subplots()
for i, data in enumerate([ga, t, l, s]):
    obj_values = [d['objective'] for d in data]
    ax.plot(np.arange(1, len(instances)+1), obj_values, label=algos[i])
ax.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Objective Value")
ax.set_title("Genetic Algorithm Convergence Plot")
plt.show()

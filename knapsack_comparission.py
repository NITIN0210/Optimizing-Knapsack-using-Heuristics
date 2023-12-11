import re
import random 
import time
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
    header_pattern = re.compile(r'knapPI_(\d+)_(\d+)_(\d+)_(\d+)$')
    #knapPI_1_100_1000_1
    #^f(\d+)_(l-d)_kp_(\d+)_(\d+)$|
    #r'^knapPI_(\d+)_(\d+)_(\d+)_(\d+)$'



    instances = []
    fnames = ['data1.txt']
    
    for fname in fnames:
        with open(fname) as f:
            lines = f.readlines()
            i = 0
            j=0
            while i < len(lines):# or 
                # search for the header of the instance
                header_match = header_pattern.match(lines[i])
                if header_match:
                    # extract the number of items and knapsack capacity from the header
                    #d1.txt group(3,4)  data1.txt group(2,3)
                    num_items = int(header_match.group(2))
                    knapsack_size = int(header_match.group(3))
                    print(num_items, knapsack_size)

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
                    #i += num_items+2 # d1.txt as extra line for data of knapsack capacity
                    i=i+num_items+1
                else:
                    # move to the next line
                    i += 1
    return instances

#file reading
instances = file_read()
print(len(instances))

def GA(instances):
    
    # Parameters
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 1000
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    ga=[]
    c=0
    for instance in instances:
        weights = instance["weights"]
        values = instance["values"]
        capacity = instance["knapsack_size"]
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

        '''# Output
        print('best solution',best_solution)
        print("Iteration: ", NUM_GENERATIONS)
        print("Time: ", elapsed_time)
        print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
        print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
        print("Objective: ", best_fitness)
        c=c+1
        
        print(c)'''
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


    # Parameters
    POPULATION_SIZE = 50
    NUM_ITERATIONS = 100
    NUM_REF_SETS = 5
    REF_SET_SIZE = 10
    s=[]
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
        '''print("Iteration: ", NUM_ITERATIONS)
        print("Time: ", elapsed_time)
        print('capacity:',capacity)
        print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
        print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
        print("Objective2 profits: ", sum([values[i] for i in range(len(best_solution)) if best_solution[i]==1]))

        print("Objective: ", best_fitness)

        c=c+1
        print(c)'''
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

    # Parameters
    NUM_ITERATIONS = 1000
    l=[]

    # Local Search
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
            # Evaluate neighbors
            neighbors = []
            for i in range(len(weights)):
                neighbor = solution[:]
                neighbor[i] = 1 - neighbor[i]
                neighbors.append(neighbor)
            neighbors_fitness = [knapsack_fitness(neighbor,weights,values,capacity) for neighbor in neighbors]

            # Select best neighbor
            best_neighbor = neighbors[0]
            best_neighbor_fitness = neighbors_fitness[0]
            for i in range(1, len(neighbors)):
                if neighbors_fitness[i] > best_neighbor_fitness:
                    best_neighbor = neighbors[i]
                    best_neighbor_fitness = neighbors_fitness[i]

            # Update solution
            solution = best_neighbor
            fitness = knapsack_fitness(solution,weights,values,capacity)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Output
        '''print("Iteration: ", NUM_ITERATIONS)
        print("Time: ", elapsed_time)
        print('capacity:',capacity)
        print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
        print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
        print("Objective2 profits: ", sum([values[i] for i in range(len(best_solution)) if best_solution[i]==1]))

        print("Objective: ", best_fitness)
        c=c+1
        print(c)'''
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

    # Parameters
    TABU_TENURE = 10
    NUM_ITERATIONS = 1000
    t=[]


    c=0
    for instance in instances:
        weights = instance["weights"]
        values = instance["values"]
        capacity = instance["knapsack_size"]
        # Tabu Search
        solution = [random.randint(0, 1) for j in range(len(weights))]
        tabu_list = [set() for j in range(len(weights))]
        best_fitness = knapsack_fitness(solution,weights,values,capacity)
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
            neighbors_fitness = [knapsack_fitness(neighbor,weights,values,capacity) for neighbor in neighbors]

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
            fitness = knapsack_fitness(solution,weights,values,capacity)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Output
        '''print("Iteration: ", NUM_ITERATIONS)
        print("Time: ", elapsed_time)
        print('capacity:',capacity)
        print("Items: ", [i+1 for i in range(len(best_solution)) if best_solution[i]==1])
        print("Weight: ", sum([weights[i] for i in range(len(best_solution)) if best_solution[i]==1]))
        print("Objective2 profits: ", sum([values[i] for i in range(len(best_solution)) if best_solution[i]==1]))

        print("Objective: ", best_fitness)
        c=c+1
        print(c)'''
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
#print(ga,t,s,l,sep='\t')
print('=========================')
for i in range(len(ga)):
    print('',ga[i],'\n',t[i],'\n',s[i],'\n',l[i])
    print('---------')


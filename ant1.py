import random
import re
class KnapsackProblem:
    def __init__(self, values, weights, capacity):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        
    def get_item_value(self, index):
        return self.values[index]
    
    def get_item_weight(self, index):
        return self.weights[index]

def ant_colony_optimization(problem, num_ants, num_iterations, alpha, beta, evaporation_rate):
    num_items = problem.num_items
    capacity = problem.capacity
    
    # initialize pheromone matrix with equal values
    pheromone_matrix = [[1.0/(num_items*capacity)]*num_items for _ in range(num_items+1)]
    
    best_value = float('-inf')
    best_items = []
    
    # iterate for given number of iterations
    for iteration in range(num_iterations):
        # initialize a list of ants
        ants = [0]*num_ants
        for i in range(num_ants):
            # initialize an ant with empty items and zero value
            ant = {'items': [], 'value': 0}
            remaining_capacity = capacity
            
            # select items according to pheromone matrix and heuristic information
            while remaining_capacity > 0 and len(ant['items']) < num_items:
                available_items = [item for item in range(num_items) if item not in ant['items']]
                
                probabilities = []
                for item in available_items:
                    p = pheromone_matrix[num_items][item]**alpha * problem.get_item_value(item)**beta
                    probabilities.append(p)
                
                total_p = sum(probabilities)
                normalized_p = [p/total_p for p in probabilities]
                
                selected_item = random.choices(available_items, weights=normalized_p)[0]
                ant['items'].append(selected_item)
                ant['value'] += problem.get_item_value(selected_item)
                remaining_capacity -= problem.get_item_weight(selected_item)
            
            # update best solution found so far
            if ant['value'] > best_value:
                best_value = ant['value']
                best_items = ant['items']
            
            ants[i] = ant
        
        # update pheromone matrix based on the ants' solutions
        for i in range(num_items):
            for j in range(num_items):
                pheromone_matrix[i][j] *= (1-evaporation_rate)
        
        for ant in ants:
            for item in ant['items']:
                pheromone_matrix[num_items][item] += 1.0/ant['value']
    
    return best_value, best_items
header_pattern = re.compile(r'^f(\d+)-(hd|ld)-kp-(\d+)-(\d+)$')

instances = []

with open('generated_instances HD-SCWith 50-500n-cap.txt') as f:
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
for instance in instances:
    weights = instance["weights"]
    values = instance["values"]
    capacity = instance["knapsack_size"]


    problem = KnapsackProblem(values, weights, capacity)

    # run ant colony optimization algorithm
    num_ants = 10
    num_iterations = 100
    alpha = 1
    beta = 2
    evaporation_rate = 0.5
    best_value, best_items = ant_colony_optimization(problem, num_ants, num_iterations, alpha, beta, evaporation_rate)

    # print the best solution found
    print("Best value:", best_value)
    print("Best items:", best_items)

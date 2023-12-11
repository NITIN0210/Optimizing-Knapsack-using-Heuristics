import re 
# regular expression to match the header of each instance
header_pattern = re.compile(r'^f(\d+)-(hd|ld)-kp-(\d+)-(\d+)$')

instances = []

with open('generated_instances HD-SC.txt') as f:
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
print('len(instances)',len(instances))
# print the information for each instance
for i, instance in enumerate(instances):
    print(f'Instance {i}:')
    print(f'Number of items: {instance["num_items"]}')
    print(f'Knapsack size: {instance["knapsack_size"]}')
    print(f'Weights: {instance["weights"]}')
    print(f'Values: {instance["values"]}')
    print()

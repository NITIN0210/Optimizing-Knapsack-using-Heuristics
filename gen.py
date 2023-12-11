import random

NUM_INSTANCES = 100

# Define functions for generating item values and weights
def generate_values(n, corr):
    values = []
    if corr == 0:  # Low dimensional uncorrelated (LD-UC)
        for i in range(n):
            values.append(random.randint(1, 100))
    else:  # High dimensional uncorrelated (HD-UC), weakly correlated (HD-WC), strongly correlated (HD-SC)
        for i in range(n):
            val = random.randint(1, 100)
            if corr == 1:  # HD-UC
                values.append(val)
            elif corr == 2:  # HD-WC
                values.append(int(val + 0.3 * random.randint(0, 100)))
            elif corr == 3:  # HD-SC
                values.append(int(val + 0.7 * random.randint(0, 100)))
    return values

def generate_weights(n, cap, corr):
    weights = []
    if corr == 0:  # Low dimensional uncorrelated (LD-UC)
        for i in range(n):
            weights.append(random.randint(1, cap))
    else:  # High dimensional uncorrelated (HD-UC), weakly correlated (HD-WC), strongly correlated (HD-SC)
        for i in range(n):
            weight = random.randint(1, cap)
            if corr == 1:  # HD-UC
                weights.append(weight)
            elif corr == 2:  # HD-WC
                weights.append(int(weight + 0.3 * random.randint(1, cap)))
            elif corr == 3:  # HD-SC
                weights.append(int(weight + 0.7 * random.randint(1, cap)))
    return weights

# Generate 100 instances
with open("generated_instances HD-SCWith 50-500n-cap.txt", "w") as f:
    for i in range(NUM_INSTANCES):
        n = random.randint(10, 50)
        cap = random.randint(50, 500)
        #corr = random.randint(0, 3)
        corr=3#0,1,2,3
        values = generate_values(n, corr)
        weights = generate_weights(n, cap, corr)
        f.write(f"f{i}-{'ld' if corr == 0 else 'hd'}-kp-{n}-{cap}\n")
        for j in range(n):
            f.write(f"{values[j]} {weights[j]}\n")
        f.write(f"{cap}\n")

import numpy as np

# Assumed user-defined function for calculating Bayes factor (replace with the actual one)
def calculate_bayes_factor(data1, data2):
    # Dummy Bayes factor function for now, to be replaced
    # You will replace this with your in-built function logic
    return np.random.uniform(0, 5)  # Example placeholder

# List of Bayes factor thresholds for stopping rules
fail_thresholds = [1.5, 1, 0.5, 1/3]
success_thresholds = [2, 3, 4, 5]

# Record of results for each combination of p_1, p_2, and Bayes factor results
results = []

# Step 1: Generate p_1 from the list [0.3, 0.35, 0.4, 0.45]
p1_list = [0.3, 0.35, 0.4, 0.45]

# Step 2: For each p_1 generate p_2 from [0.97*p_1, 0.99*p_1, p_1, 1.01*p_1, 1.03*p_1, 1.05*p_1]
p2_multiplier_list = [0.97, 0.99, 1.00, 1.01, 1.03, 1.05]

# Step 3: Sample 10,000 samples each day from Bernoulli(p_1) and Bernoulli(p_2)
n_samples = 10000
n_days = 50

# Loop over each p_1
for p1 in p1_list:
    # Generate corresponding p_2 values
    p2_list = [multiplier * p1 for multiplier in p2_multiplier_list]

    # Loop over each p_2
    for p2 in p2_list:
        # Simulate data for 50 days
        bayes_factors = []
        conclusion = None
        for day in range(n_days):
            # Generate 10,000 samples from Bernoulli(p_1) and Bernoulli(p_2)
            data1 = np.random.binomial(1, p1, n_samples)
            data2 = np.random.binomial(1, p2, n_samples)

            # Step 5: Fit the Bayes factor calculation to the data
            bayes_factor = calculate_bayes_factor(data1, data2)
            bayes_factors.append(bayes_factor)

            # Step 6: Check stopping conditions
            if any(bayes_factor >= t for t in success_thresholds):
                conclusion = 'SUCCESS'
                break
            elif any(bayes_factor <= t for t in fail_thresholds):
                conclusion = 'FAIL'
                break

        # Record the results
        results.append({
            'p1': p1,
            'p2': p2,
            'day': day + 1,  # Day when the test stopped
            'bayes_factor': bayes_factors[-1],  # Final Bayes factor at stopping
            'conclusion': conclusion or 'NO_STOP'  # Conclusion after the final day
        })

# Display the results
for result in results:
    print(f"p1: {result['p1']}, p2: {result['p2']}, day: {result['day']}, "
          f"bayes_factor: {result['bayes_factor']:.2f}, conclusion: {result['conclusion']}")

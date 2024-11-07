inputs = [1.1, 2.2, 3.3]

weights = [
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
]

bias = [0, 0, 1]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, bias):
    neuron_output = 0
    
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
        
    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)
print(layer_outputs)

import numpy as np

inputs = np.array([1.1, 2.2, 3.3])

weights = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

bias = np.array([0, 0, 1])

layer_outputs = np.dot(inputs, weights.T) + bias

print(layer_outputs)
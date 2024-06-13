import random
import math

def breed_neural_networks(list_of_neural_networks, successful_percentage, layer_step=1, neuron_step=5):
    total_neural_networks = len(list_of_neural_networks)
    successful_neural_networks_count = max([2, math.ceil(total_neural_networks * successful_percentage / 100)])
    successful_neural_networks = list_of_neural_networks[0:successful_neural_networks_count]
    remaining_neural_networks = total_neural_networks - successful_neural_networks_count
    
    new_neural_networks = []
    for a in range(remaining_neural_networks):
        neurals = random.choices(successful_neural_networks, k=2)
        new_neural_network = mix_neural_networks(neurals[0], neurals[1], layer_step=layer_step, neuron_step=neuron_step)
        new_neural_networks.append(new_neural_network)
    
    final_neural_networks = []
    for network in new_neural_networks: final_neural_networks.append(network)
    for network in successful_neural_networks: final_neural_networks.append(network)
    return final_neural_networks, remaining_neural_networks

def mix_neural_networks(neural_network_a, neural_network_b, layer_step=1, neuron_step=5):
    neural_network_a_b = []
    layer_count = 0
    layer_min = min([len(neural_network_a), len(neural_network_b)])
    layer_max = max([len(neural_network_a), len(neural_network_b)])

    layer_count_min = layer_min - layer_step
    layer_count_max = layer_max + layer_step

    layer_count = random.randrange(max([1, layer_count_min]), layer_count_max + 1)

    for layer in range(layer_count):
        if layer < len(neural_network_a) and layer < len(neural_network_b):
            neuron_min = min([neural_network_a[layer], neural_network_b[layer]]) - neuron_step
            neuron_max = max([neural_network_a[layer], neural_network_b[layer]]) + neuron_step
        elif layer < len(neural_network_a):
            neuron_min = neural_network_a[layer] - neuron_step
            neuron_max = neural_network_a[layer] + neuron_step
        elif layer < len(neural_network_b):
            neuron_min = neural_network_b[layer] - neuron_step
            neuron_max = neural_network_b[layer] + neuron_step
        else:
            neuron_min = 1
            neuron_max = neuron_step * 2
        neuron_count = random.randrange(max([1, neuron_min]), neuron_max + 1)
        neural_network_a_b.append(neuron_count)
    return neural_network_a_b
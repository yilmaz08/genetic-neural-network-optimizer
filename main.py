import mixer
import optimizer
import tensorflow as tf

# print(tf.config.list_physical_devices('GPU'))
# exit()


# Datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# Variables
input_shape=(28,28)
output_neurons=10
epochs = 10
network_count = 20
neural_networks = []
successful_percentage = 25
default_network = [128,128,128]
layer_step = 2
neuron_step = 10
# Generate black networks
for a in range(network_count): neural_networks.append(default_network)
# Randomize the networks (to mix networks)
for a in range(2): neural_networks, news = mixer.breed_neural_networks(neural_networks, successful_percentage, layer_step, neuron_step)
# LOOP
while True:
    new_neural_networks_with_accuracy = []
    # Optimize and Evaluate
    print("--- OPTIMIZE & EVALUATE ---")
    for network in neural_networks:
        name = f"model{epochs}e"
        for a in network: name += f"_{a}"
        print(network)
        new_name, accuracy, loss = optimizer.optimize_and_evaluate(
            network=network,
            new_name=name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_shape=input_shape,
            output_neurons=output_neurons,
            epochs=epochs
        )
        new_neural_networks_with_accuracy.append((accuracy, network))
    # Order by results
    new_neural_networks = []
    ordered_new_neural_networks_with_accuracy = sorted(new_neural_networks_with_accuracy, key= lambda x: x[0], reverse=True)
    for network in ordered_new_neural_networks_with_accuracy: new_neural_networks.append(network[1])
    print("NNs were ordered")
    # Breed
    print("--- BREED ---")
    neural_count = len(neural_networks)
    neural_networks, new_neurals = mixer.breed_neural_networks(new_neural_networks, successful_percentage, layer_step, neuron_step)
    print(f"{neural_count} NNs -> {len(neural_networks)} NNs | {new_neurals} New {len(neural_networks) - new_neurals} Same")
    print(neural_networks)
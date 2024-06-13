import tensorflow as tf
import numpy as np
import os

def optimize(network, x_train, y_train, epochs, input_shape, output_neurons):
    # Create
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    for neuron_count in network:
        model.add(tf.keras.layers.Dense(neuron_count, activation='relu'))

    model.add(tf.keras.layers.Dense(output_neurons, activation='softmax'))
    # Compile
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Train
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    return model

def evaluate(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy

def optimize_and_evaluate(network, new_name, x_train, y_train, x_test, y_test, input_shape, output_neurons, epochs):
    # new_name = f"model{epochs}e{name}"
    if os.path.exists(f"models/{new_name}.keras"):
        _model = tf.keras.models.load_model(f"models/{new_name}.keras")
        print(f"{new_name} was loaded")
    else:
        _model = optimize(network=network, x_train=x_train, y_train=y_train, epochs=epochs, input_shape=input_shape, output_neurons=output_neurons)
        print(f"{new_name} was optimized")
    # Evaluate
    loss, accuracy = evaluate(model=_model, x_test=x_test, y_test=y_test)
    print(f"{new_name} was evaluated. Accuracy: {accuracy} Loss: {loss}")
    # Save
    if not os.path.exists(f"models/{new_name}.keras"):
        _model.save(f"models/{new_name}.keras")
        print(f"{new_name} was saved")

    return new_name, accuracy, loss
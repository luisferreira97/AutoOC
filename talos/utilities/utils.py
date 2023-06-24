from os import path

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def reverse(data_list):
    length = len(data_list)
    s = length
    new_list = [None] * length
    for item in data_list:
        s = s - 1
        new_list[s] = item
    return new_list


def get_model_from_encoder(encoder):
    # get model configs
    configs = encoder.get_config()
    # extract layers
    layers = configs.get("layers")
    # remove layers' names to prevent errors
    for layer in layers:
        if layer["config"]["name"] not in ["latent", "input"]:
            del layer["config"]["name"]
    # reverse layers (last one is the latent space, should not be repeated!)
    rev_layers = reverse(layers[1:-1])
    # concat layers
    all_layers = layers + rev_layers
    # create model
    ae = Sequential.from_config(all_layers)

    return ae


def get_model_from_encoder2(encoder):
    # get model configs
    configs = encoder.get_config()
    # extract layers
    layers = configs.get("layers")
    # get input layer shape
    current_shape = layers[0]["config"]["batch_input_shape"][1]
    # adjust layers' shapes
    for layer in layers:
        if "units" in layer["config"]:
            current_shape = int(
                current_shape * (layer["config"]["units"] / 100))
            if current_shape < 1:
                current_shape = 1
            layer["config"]["units"] = current_shape
            print(layer["config"]["units"] / 10)
    # remove layers' names to prevent errors
    for layer in layers:
        if layer["config"]["name"] not in ["latent", "input"]:
            del layer["config"]["name"]
    # reverse layers (last one is the latent space, should not be repeated!)
    rev_layers = reverse(layers[1:-1])
    # concat layers
    all_layers = layers + rev_layers
    # create model
    ae = Sequential.from_config(all_layers)
    return ae

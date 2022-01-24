"""
Skeleton model(s) to make model creation easier
"""
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras import optimizers


# For Neural Networks, sequential
def build_model_neural_sequential(
        size, # number of layers
        activation, # what activation function to use in the Dense layers
        weight_regularizer,
        dropout,
        batch_norm,
        loss_function,
        optimizer,
        metrics):
    lay = [Flatten(input_shape=(150, 84))]  # Input layer

    numb_layers = size
    numb_neurons = size ** 2 * 4 # Can this be a different value?

    for i in range(numb_layers):
        lay.append(Dense(
            numb_neurons,
            activation=activation,
            kernel_regularizer=weight_regularizer))

        if batch_norm:
            lay.append(BatchNormalization())
        if dropout:
            lay.append(Dropout(0.2))

        # Output layer:
        # nb_classes is 11 for ModelD and ModelY
        # nb_classes is 2 for Model CC
        # Which activation function to use, the same for all three or different??
        lay.append(Dense('nb_classes', 'activation'))

        model = models.Sequential(lay)
        model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=metrics
        )
        model.summary()
        return model

def build_model_CC():



    return modelCC

def build_model_D():

    return modelD

def build_model_Y():

    return modelY

def merged_CCDY():

     return modelCCDY


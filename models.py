"""
Skeleton model(s) to make model creation easier
- Random Forest
- Boosting
- Wrapper of Random Forest and Boosting
"""
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras import optimizers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# For Neural Networks, sequential (From lecture slides "optimization and regularization"
def build_model_neural_sequential(
        size,  # number of layers
        activation,  # what activation function to use in the Dense layers
        weight_regularizer,
        dropout,
        batch_norm,
        loss_function,
        optimizer,
        metrics):
    lay = [Flatten(input_shape=(150, 84))]  # Input layer

    numb_layers = size
    numb_neurons = size ** 2 * 4  # Can this be a different value?

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

# Random forest, there are more parameters to consider in the future for optimization in all the models.
def build_random_forest(number_estimators,
                        criterion,
                        max_depth,
                        min_samples_split,
                        min_samples_leaf,
                        max_features,
                        bootstrap,
                        random_state
                        ):
    rf_model = RandomForestClassifier(n_estimators=number_estimators,
                                      criterion=criterion,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      max_features=max_features,
                                      bootstrap=bootstrap,
                                      random_state=random_state)

    return rf_model

# Gradient Boosting Classifier, what about other boosting`s? XGBoost, AdaBoost etc.
def build_boosting(loss,
                   learning_rate,
                   n_estimators,
                   subsample,
                   ):
    gb_model = GradientBoostingClassifier(loss=loss,
                                          learning_rate=learning_rate,
                                          n_estimators=n_estimators,
                                          subsample=subsample)

    return gb_model


# Wrapping the Random Forest and Boosting models
# Boosting first, then take the final adjusted dataset(the estimators) and insert it into a Random Forest Model:
# After training the boosting model, call this function and use the returned model to train the random forest??
def merge_rf_boost(
        rf_model,
        b_model
        ):
    b_model.estimators_ += rf_model.estimators_
    b_model.n_estimators = len(b_model.estimators_)

    return b_model


def get_model_CC(model):
    current_CC_model = model
    return current_CC_model


def get_model_D(model):
    current_D_model = model
    return current_D_model


def get_model_Y(model):
    current_Y_model = model
    return current_Y_model

# Sequence accuracy: prediction is correct when all three sub-models are making correct predictions (1 point),
# Character accuracy: prediction is 1/3 correct (1/3 point) if one sub model is making correct predictions,
# in addition to standard accuracy measures.
def combined_predictions(cc_predict, d_predict, y_predict):
    seq_score = cc_predict + d_predict + y_predict
    char_predict = 0
    return

"""
Skeleton model(s) to make model creation easier
- Random Forest
- Boosting
- Wrapper of Random Forest and Boosting
"""
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Inspired by the resnet architecture
input_layer = Input(shape=(84, 150, 3))


def conv(input_layer, filters=64, kernel_size=(3, 3), pool_size=(2, 2), dropout_rate=0.2):
    x = Conv2D(filters=filters, padding='same', kernel_size=kernel_size)(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    return SpatialDropout2D(dropout_rate)(x)


def residualblock(origin_layer):
    # First block
    l1 = Conv2D(filters=64, padding='same', kernel_size=(3, 3))(origin_layer)
    l1 = BatchNormalization()(l1)
    l1 = ReLU()(l1)

    # Second block
    l2 = Conv2D(filters=64, padding='same', kernel_size=(3, 3))(l1)
    l2 = BatchNormalization()(l2)
    l2 = ReLU()(l2)

    # Third block
    l3 = Conv2D(filters=64, padding='same', kernel_size=(3, 3))(l2)
    l3 = BatchNormalization()(l3)
    l3 = ReLU()(l3)

    # Short cut
    short_cut = Conv2D(filters=64, padding='same', kernel_size=(3, 3))(origin_layer)
    short_cut = BatchNormalization()(short_cut)

    return SpatialDropout2D(0.2)(
        Concatenate(axis=-1)([l3, origin_layer])
    )


def build_cnn_model_resnet(ccmodel, dmodel, ymodel):
    CCMODEL = ccmodel
    DMODEL = dmodel
    YMODEL = ymodel
    edges_layer = conv(input_layer, filters=32, kernel_size=(5, 5))
    x = conv(edges_layer)
    x = residualblock(x)
    x = MaxPooling2D(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)
    x = Dense(1 if CCMODEL else 11, activation='sigmoid' if CCMODEL else 'softmax')(x)

    opt = Adam(learning_rate=0.001)
    if CCMODEL:
        cc_model = Model(input_layer, x, name='CC model')
        cc_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
        cc_model.summary()
        return cc_model
    elif DMODEL:
        d_model = Model(input_layer, x, name='D model')
        d_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        d_model.summary()
        return d_model
    elif YMODEL:
        y_model = Model(input_layer, x, name='Y model')
        y_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        y_model.summary()
        return y_model


def build_cnn_model_default(ccmodel, dmodel, ymodel):
    CCMODEL = ccmodel
    DMODEL = dmodel
    YMODEL = ymodel
    input = input_layer
    x = Conv2D(filters=32, padding='same', kernel_size=(3, 3))(input)
    x = Conv2D(filters=32, padding='same', kernel_size=(3, 3))(x)
    x = Conv2D(filters=32, padding='same', kernel_size=(3, 3))(x)
    x = Flatten()(x)
    x = Dense(1 if CCMODEL else 11, activation='sigmoid' if CCMODEL else 'softmax')(x)

    opt = Adam(learning_rate=0.001)
    if CCMODEL:
        cc_model = Model(input_layer, x, name='CC model')
        cc_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
        cc_model.summary()
        return cc_model
    elif DMODEL:
        d_model = Model(input_layer, x, name='D model')
        d_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        d_model.summary()
        return d_model
    elif YMODEL:
        y_model = Model(input_layer, x, name='Y model')
        y_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        y_model.summary()
        return y_model

import tensorflow as tf
import numpy as np
import pickle

from keras.models import Model
from keras.layers import (
    Activation,
    Dense,
    Input,
    Conv1D,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Dropout,
    SpatialDropout1D
)


def residual_block_2019(x, i, activation, num_filters, kernel_size, padding, dropout_rate=0.1, name=''):
    # name of the layer
    name = name + '_dilation_%d' % i
    # 1x1 conv. of input (so it can be added as residual)
    res_x = Conv1D(num_filters, 1, padding='same', name=name + '_1x1_conv_residual')(x)
    # one dilated convolution with dilation rates of i (Bock 2019)
    conv_dilated = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + '_dilated_conv')(x)
    # apply activation function
    x = Activation(activation, name=name + '_activation')(conv_dilated)
    # apply spatial dropouts
    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_%f' % dropout_rate)(x)
    # 1x1 conv. to obtain a representation with the same size as the residual
    x = Conv1D(num_filters, 1, padding='same', name=name + '_1x1_conv')(x)
    # add the residual to the processed data and also return it as skip connection
    return tf.keras.layers.add([res_x, x], name=name + '_merge_residual'), x


class TCN:
    def __init__(self,
                 num_filters=20,
                 kernel_size=5,
                 dilations=None,
                 activation='elu',
                 padding='same',
                 dropout_rate=0.1,
                 name='tcn'):
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.name = name
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_filters = [num_filters] * len(dilations)
        self.padding = padding

    def __call__(self, inputs):
        x = inputs
        # gather skip connections, each having a different context
        skip_connections = []
        # build the TCN models
        for i, num_filters in zip(self.dilations, self.num_filters):
            # feed the output of the previous layer into the next layer
            # increase dilation rate for each consecutive layer
            x, skip_out = residual_block_2019(
                x, i, self.activation, num_filters, self.kernel_size, self.padding, self.dropout_rate, name=self.name)
            # collect skip connection
            skip_connections.append(skip_out)
        # here, the linear activation is redundant, however, for the sake of simplicity, the layer is kept as it is
        # in Bock 2020, the ELU is used instead
        x = Activation('linear', name=self.name + '_activation')(x)
        # merge the skip connections by simply adding them
        skip = tf.keras.layers.add(skip_connections, name=self.name + '_merge_skip_connections')
        return x, skip


def create_2019_model_(input_shape, num_filters=16, num_dilations=11, kernel_size=5, activation='elu', dropout_rate=0.1, excption=None):
    # input layer
    input_layer = Input(shape=input_shape)
    # stack of 3 conv layers, each conv, activation, max. pooling & dropout
    conv_1 = Conv2D(num_filters, (3, 3), padding='valid', name='conv_1_conv')(input_layer)
    conv_1 = Activation(activation, name='conv_1_activation')(conv_1)
    conv_1 = MaxPooling2D((1, 3), name='conv_1_max_pooling')(conv_1)
    conv_1 = Dropout(dropout_rate, name='conv_1_dropout')(conv_1)

    conv_2 = Conv2D(num_filters, (3, 3), padding='valid', name='conv_2_conv')(conv_1)
    conv_2 = Activation(activation, name='conv_2_activation')(conv_2)
    conv_2 = MaxPooling2D((1, 3), name='conv_2_max_pooling')(conv_2)
    conv_2 = Dropout(dropout_rate, name='conv_2_dropout')(conv_2)
    if excption:
        conv_3 = Conv2D(num_filters, (1, 8), padding='valid', name='conv_3_conv')(conv_2)
    else:
        conv_3 = Conv2D(num_filters, (1, 8), padding='valid', name='conv_3_conv')(conv_2)
    conv_3 = Activation(activation, name='conv_3_activation')(conv_3)
    conv_3 = Dropout(dropout_rate, name='conv_3_dropout')(conv_3)
    # reshape layer to reduce dimensions
    x = Reshape((-1, num_filters), name='tcn_input_reshape')(conv_3)
    # TCN layers
    dilations = [2 ** i for i in range(num_dilations)]
    tcn, skip = TCN(
        num_filters=num_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        activation=activation,
        padding='same',
        dropout_rate=dropout_rate,
    )(x)

    # beats
    beats = Dropout(dropout_rate, name='beats_dropout')(tcn)
    beats = Dense(1, name='beats_dense')(beats)
    beats = Activation('sigmoid', name='beats')(beats)

    # tempo
    # tempo = Dropout(dropout_rate, name='tempo_dropout')(skip)
    # tempo = GlobalAveragePooling1D(name='tempo_global_average_pooling')(tempo)
    # GaussianNoise layer is not mentioned in Bock 2019 article -- it causes wrong predictions of beats, which is weird,
    # because beats and tempo should not interfere together, maybe bug in Keras?
    # tempo = GaussianNoise(dropout_rate, name='tempo_noise')(tempo)
    # tempo = Dense(300, name='tempo_dense')(tempo)
    # tempo = Activation('softmax', name='tempo')(tempo)
    return Model(input_layer, outputs=beats)


def create_2019_model(input_shape, load_weights=True):
    if input_shape == (None, 74, 1):
        mdl = create_2019_model_(input_shape=input_shape, excption=True)
    else:
        mdl = create_2019_model_(input_shape=input_shape)
    if load_weights:
        with open('data/beats_tcn_1.pkl', 'rb') as f:
            madmom_weights = pickle.load(f)

        # Input CNN weights and biases
        # all channels needs to be flipped on both sides, because madmom uses 2D convolution
        # Keras and Torch use 2D correlation, because flipping of kernels in NNs is not necessary
        # furthermore, dimensions need to be flipped so the weights of the kernel can be loaded by Keras
        # biases in madmom are the same as in Keras, they can be loaded in the same way

        # First layer: 3x3 conv
        conv2d_1_weights = np.flip(np.flip(np.transpose(madmom_weights.layers[0].weights, (2, 3, 0, 1)), 0), 1)
        conv2d_1_bias = madmom_weights.layers[0].bias
        # set weights and bias
        mdl.layers[1].set_weights([conv2d_1_weights, conv2d_1_bias])

        # Second layer: 3x3 conv
        conv2d_2_weights = np.flip(np.flip(np.transpose(madmom_weights.layers[2].weights, (2, 3, 0, 1)), 0), 1)
        conv2d_2_bias = madmom_weights.layers[2].bias
        # set weights and bias
        mdl.layers[5].set_weights([conv2d_2_weights, conv2d_2_bias])

        # Third layer: 1x8 conv
        conv2d_3_weights = np.flip(np.flip(np.transpose(madmom_weights.layers[4].weights, (2, 3, 0, 1)), 0), 1)
        conv2d_3_bias = madmom_weights.layers[4].bias
        # set weights and bias
        mdl.layers[9].set_weights([conv2d_3_weights, conv2d_3_bias])

        # Dense layers weights
        # vahy dense vrstev se muzou rovnou nacist bez uprav
        # beats dense layer
        beats_dense_weights = madmom_weights.layers[7].layers[0].weights
        beats_dense_bias = madmom_weights.layers[7].layers[0].bias
        mdl.layers[81].set_weights([beats_dense_weights, beats_dense_bias])

        # tempo dense layer
        # tempo_dense_weights = madmom_weights.layers[7].layers[1].layers[1].weights
        # tempo_dense_bias = madmom_weights.layers[7].layers[1].layers[1].bias
        # mdl.layers[85].set_weights([tempo_dense_weights, tempo_dense_bias])

        # TCN blocks weights
        if input_shape == (None, 74, 1):
            dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        start_layer = 13
        # TCN madmom weights
        TCN_blocks_weights = madmom_weights.layers[6].tcn_blocks
        for i in range(len(dilations)):
            conv_layer_indices = [start_layer, start_layer + 3, start_layer + 4]
            # dilated conv
            # flipping the kernel again, this time only in one of the axis, because it is 1D convolution
            # at the same time we flip the dimensions, so it fits the Keras standard
            # weights have one redundant dimension compared to the Keras so we squeeze it
            dilated_conv_weights = np.transpose(np.flip(TCN_blocks_weights[i].dilated_conv.weights, 3),
                                                (3, 0, 1, 2)).squeeze()
            dilated_conv_bias = TCN_blocks_weights[i].dilated_conv.bias
            mdl.layers[conv_layer_indices[0]].set_weights([dilated_conv_weights, dilated_conv_bias])
            # residual conv
            # these weights can be loaded, but we need to add 0. dimension, so it fits Keras
            residual_conv_weights = np.expand_dims(TCN_blocks_weights[i].residual_conv.weights, axis=0)
            residual_conv_bias = TCN_blocks_weights[i].residual_conv.bias
            mdl.layers[conv_layer_indices[1]].set_weights([residual_conv_weights, residual_conv_bias])
            # skip connection conv
            # again can be loaded, but one dimension is added
            skip_conv_weights = np.expand_dims(TCN_blocks_weights[i].skip_conv.weights, axis=0)
            skip_conv_bias = TCN_blocks_weights[i].skip_conv.bias
            mdl.layers[conv_layer_indices[2]].set_weights([skip_conv_weights, skip_conv_bias])
            # increment layer idx
            start_layer += 6
    return mdl

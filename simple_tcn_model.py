import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Activation,
    Dense,
    Input,
    Conv1D,
    # Conv2D,
    # MaxPooling2D,
    Reshape,
    Dropout,
    SpatialDropout1D,
    # GaussianNoise,
    # GlobalAveragePooling1D
)

import keras
in_shape = (None, 81, 1)


def residual_block_simple(x, i, activation, num_filters, kernel_size, padding, dropout_rate=0.1, name=''):
    name = name + '_dilation_%d' % i
    if i == 1:
        res_x = Conv1D(filters=num_filters, kernel_size=1, name='conv_freq_reduction')(x)
    else:
        res_x = x
    # first dilated conv
    conv_dilated = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + '_dilated_conv_1')(x)
    x = Activation(activation, name=name + '_activation_1')(conv_dilated)
    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_1')(x)
    # second dilated conv
    conv_dilated_2 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + '_dilated_conv_2')(x)
    x = Activation(activation, name=name + '_activation_2')(conv_dilated_2)
    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_2')(x)
    # add the residual to the processed data
    return tf.keras.layers.add([res_x, x], name=name + '_merge_residual'), x


def residual_block_gated(x, i, activation, num_filters, kernel_size, padding, dropout_rate=0.1, name=''):
    name = name + '_dilation_%d' % i
    if i == 1:
        res_x = Conv1D(filters=num_filters, kernel_size=1, name='conv_freq_reduction')(x)
    else:
        res_x = x
    block_in = x
    # first dilated conv
    conv_dilated = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + '_dilated_conv_1')(block_in)
    conv1_activated = Activation('tanh', name=name + '_activation_1')(conv_dilated)
    # second dilated conv
    conv_dilated_2 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + '_dilated_conv_2')(block_in)
    conv2_activated = Activation('sigmoid', name=name + '_activation_2')(conv_dilated_2)
    gated_activation = keras.layers.multiply([conv1_activated, conv2_activated])
    gated_dropout = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_1')(gated_activation)
    x = Conv1D(num_filters, 1, padding='same', name=name + '_1x1_conv')(gated_dropout)
    # add the residual to the processed data
    return tf.keras.layers.add([res_x, x], name=name + '_merge_residual'), x


def residual_block_2020(x, i, activation, num_filters, kernel_size, padding, dropout_rate=0.1, name=''):
    name = name + '_dilation_%d' % i
    # 1x1 conv. of input (so it can be added as residual)
    res_x = Conv1D(num_filters, 1, padding='same', name=name + '_1x1_conv_residual')(x)
    # two dilated convolutions, with dilation rates of i and 2i
    conv_1 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + '_dilated_conv_1')(x)
    conv_2 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i * 2,
        padding=padding,
        name=name + '_dilated_conv_2')(x)
    # concatenate the output of the two dilations
    concat = tf.keras.layers.concatenate([conv_1, conv_2], name=name + '_concat')
    # apply activation function
    x = Activation(activation, name=name + '_activation')(concat)
    # apply spatial dropout
    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_%f' % dropout_rate)(x)
    # 1x1 conv. to obtain a representation with the same size as the residual
    x = Conv1D(num_filters, 1, padding='same', name=name + '_1x1_conv')(x)
    # add the residual to the processed data and also return it as skip connection
    return tf.keras.layers.add([res_x, x], name=name + '_merge_residual'), x


class TCN_simple:
    def __init__(self,
                 num_filters=16,
                 kernel_size=5,
                 dilations=None,
                 activation='elu',
                 padding='same',
                 dropout_rate=0.1,
                 name='tcn',
                 version='2020'):
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.name = name
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_filters = [num_filters] * len(dilations)
        self.padding = padding
        self.version = version

    def __call__(self, inputs):
        x = inputs
        # build the TCN models
        skip_connections = []
        for i, num_filters in zip(self.dilations, self.num_filters):
            # feed the output of the previous layer into the next layer
            # increase dilation rate for each consecutive layer
            if self.version == '2020':
                x, skip_out = residual_block_2020(
                    x, i,
                    self.activation,
                    num_filters,
                    self.kernel_size,
                    self.padding,
                    self.dropout_rate,
                    name=self.name)
                skip_connections.append(skip_out)
            elif self.version == 'gated':
                x, skip_out = residual_block_gated(
                    x, i,
                    self.activation,
                    num_filters,
                    self.kernel_size,
                    self.padding,
                    self.dropout_rate,
                    name=self.name)
                skip_connections.append(skip_out)
            else:
                x, skip_out = residual_block_simple(
                    x, i,
                    self.activation,
                    num_filters,
                    self.kernel_size,
                    self.padding,
                    self.dropout_rate,
                    name=self.name)
                skip_connections.append(skip_out)
        x = Activation(activation=self.activation, name=self.name + '_activation')(x)
        skip = keras.layers.concatenate(skip_connections, name='skip_concat')
        return x, skip



def create_simple_tcn(input_shape, num_filters=20, num_dilations=11, kernel_size=5, activation='elu', dropout_rate=0.1,
                      skip_output=True, version='gated'):
    # default: num_filters=25, num_dilations=11, kernel_size=7
    # stepna: num_filters=20, num_dilations=11, kernel_size=5

    # input layer
    input_layer = Input(shape=input_shape)
    # swap spectrogram dimensions
    x = Reshape((-1, input_shape[1]), name='tcn_input_reshape')(input_layer)
    # TCN layers
    dilations = [2 ** i for i in range(num_dilations)]
    tcn, skip = TCN_simple(
        num_filters=num_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        activation=activation,
        padding='same',
        dropout_rate=dropout_rate,
        version=version
    )(x)
    if skip_output:
        mixer = Conv1D(filters=1, kernel_size=1, name='mixer')(skip)
        dropout = Dropout(rate=dropout_rate)(mixer)
        beats = Dense(units=1, activation='sigmoid')(dropout)
    else:
        dropout = Dropout(rate=dropout_rate)(tcn)
        beats = Dense(units=1, activation='sigmoid')(dropout)
    return Model(input_layer, outputs=beats)

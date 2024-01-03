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
    SpatialDropout1D,
    # GaussianNoise,
    # GlobalAveragePooling1D,
)
import keras


def residual_block_2020_(x, i, activation, num_filters, kernel_size, padding, dropout_rate=0.1, name=''):
    # name of the layer
    name = name + '_dilation_%d' % i
    # 1x1 conv. of input (so it can be added as residual)
    res_x = Conv1D(num_filters, 1, padding='same', name=name + '_1x1_conv_residual')(x)
    # two dilated convolutions, with dilation rates of i and 2i
    conv_1 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i,
        padding=padding,
        name=name + '_dilated_conv_1',
    )(x)
    conv_2 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=i * 2,
        padding=padding,
        name=name + '_dilated_conv_2',
    )(x)
    # concatenate the output of the two dilations
    concat = keras.layers.concatenate([conv_1, conv_2], name=name + '_concat')
    # apply activation function
    x = Activation(activation, name=name + '_activation')(concat)
    # apply spatial dropout
    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_%f' % dropout_rate)(x)
    # 1x1 conv. to obtain a representation with the same size as the residual
    x = Conv1D(num_filters, 1, padding='same', name=name + '_1x1_conv')(x)
    # add the residual to the processed data and also return it as skip connection
    return keras.layers.add([res_x, x], name=name + '_merge_residual'), x


class TCN_2020:
    def __init__(
            self,
            num_filters=20,
            kernel_size=5,
            dilations=None,
            activation='elu',
            padding='same',
            dropout_rate=0.15,
            name='tcn',
    ):
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.name = name
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_filters = [num_filters] * len(dilations)
        self.padding = padding

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

    def __call__(self, inputs):
        x = inputs
        # gather skip connections, each having a different context
        skip_connections = []
        # build the TCN models
        for i, num_filters in zip(self.dilations, self.num_filters):
            # feed the output of the previous layer into the next layer
            # increase dilation rate for each consecutive layer
            x, skip_out = residual_block_2020_(
                x, i, self.activation, num_filters, self.kernel_size, self.padding, self.dropout_rate, name=self.name
            )
            # collect skip connection
            skip_connections.append(skip_out)
        # activate the output of the TCN stack
        x = Activation(self.activation, name=self.name + '_activation')(x)
        # merge the skip connections by simply adding them
        skip = keras.layers.add(skip_connections, name=self.name + '_merge_skip_connections')
        return x, skip


def create_2020_model(input_shape, num_filters=20, num_dilations=11, kernel_size=5, activation='elu', dropout_rate=0.1):
    # input layer
    input_layer = Input(shape=input_shape)

    # stack of 3 conv layers, each conv, activation, max. pooling & dropout
    conv_1 = Conv2D(num_filters, (3, 3), padding='valid', name='conv_1_conv')(input_layer)
    conv_1 = Activation(activation, name='conv_1_activation')(conv_1)
    conv_1 = MaxPooling2D((1, 3), name='conv_1_max_pooling')(conv_1)
    conv_1 = Dropout(dropout_rate, name='conv_1_dropout')(conv_1)

    conv_2 = Conv2D(num_filters, (1, 10), padding='valid', name='conv_2_conv')(conv_1)
    conv_2 = Activation(activation, name='conv_2_activation')(conv_2)
    conv_2 = MaxPooling2D((1, 3), name='conv_2_max_pooling')(conv_2)
    conv_2 = Dropout(dropout_rate, name='conv_2_dropout')(conv_2)

    conv_3 = Conv2D(num_filters, (3, 3), padding='valid', name='conv_3_conv')(conv_2)
    conv_3 = Activation(activation, name='conv_3_activation')(conv_3)
    conv_3 = MaxPooling2D((1, 3), name='conv_3_max_pooling')(conv_3)
    conv_3 = Dropout(dropout_rate, name='conv_3_dropout')(conv_3)

    # reshape layer to reduce dimensions
    x = Reshape((-1, num_filters), name='tcn_input_reshape')(conv_3)

    # TCN layers
    dilations = [2 ** i for i in range(num_dilations)]
    tcn, skip = TCN_2020(
        num_filters=num_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        activation=activation,
        padding='same',
        dropout_rate=dropout_rate,
    )(x)

    # output layers; beats & downbeats use TCN output, tempo the skip connections
    beats = Dropout(dropout_rate, name='beats_dropout')(tcn)
    beats = Dense(1, name='beats_dense')(beats)
    beats = Activation('sigmoid', name='beats')(beats)

    # instantiate a Model and return it
    return Model(input_layer, outputs=beats)

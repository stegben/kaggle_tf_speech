import tensorflow as tf

from ..names import (
    OP_INFERENCE,
    OP_LOSS,
    OP_TRAIN,
)
from .common import (
    ACTIVATIONS,
    get_input,
    dense_1d_block,
)


def build_conv_1d_dense_net(
        input_dim,
        output_dim,
        dense_net_structure,
        dense_structure,
        do_global_pooling=False,
        log_epsilon=None,
        seed_base=2017,
    ):
    x_place, y_place, sample_weight_place, lr_place, _, conv_dropout_place, dense_dropout_place, is_training = get_input(input_dim, output_dim)
    if log_epsilon is not None:
        x_place = tf.log(tf.abs(x_place) + log_epsilon)
    x_place_reshape = tf.expand_dims(x_place, axis=1)
    print(x_place_reshape.shape)

    dense_conv_output = x_place_reshape
    for idx_dense_block, (
            n_layers,
            n_kernels,
            n_compressed_kernels,
            window_length,
            activation,
            gated,
            pool_length,
        ) in enumerate(dense_net_structure):
        output_wave = dense_1d_block(
            input_wave=dense_conv_output,
            n_layers=n_layers,
            n_kernels=n_kernels,
            n_compressed_kernels=n_compressed_kernels,
            window_length=window_length,
            activation=activation,
            gated=gated,
            name='{}_dense_1d_net'.format(idx_dense_block),
        )
        print(output_wave.shape)
        output_wave_pooled = tf.transpose(tf.layers.average_pooling1d(
            tf.transpose(output_wave, perm=[0, 2, 1]),
            pool_size=(pool_length,),
            strides=(pool_length,),
            data_format='channels_first',
            name="dense_1d_pool_{}".format(idx_dense_block),
        ), perm=[0, 2, 1])
        print(output_wave_pooled.shape)
        dense_conv_output = tf.nn.dropout(
            output_wave_pooled,
            keep_prob=(1 - conv_dropout_place),
        )
    print(dense_conv_output.shape)
    if do_global_pooling:
        dense_conv_output = tf.transpose(tf.layers.average_pooling1d(
            tf.transpose(dense_conv_output, perm=[0, 2, 1]),
            pool_size=(dense_conv_output.shape[2],),
            strides=(dense_conv_output.shape[2],),
            data_format='channels_first',
            name="global_pool",
        ), perm=[0, 2, 1])
        print(dense_conv_output.shape)

    dense_input = tf.layers.flatten(dense_conv_output)
    print(dense_input.shape)
    a = dense_input
    dense_input_dim = a.shape[1]
    for n_layer, (n_neuron, activation) in enumerate(dense_structure):
        dense_output_dim = n_neuron
        weights = tf.get_variable(
            'weights_{}'.format(n_layer + 1),
            shape=[dense_input_dim, dense_output_dim],
            initializer=tf.keras.initializers.lecun_uniform(seed=seed_base + n_layer),
        )
        biases = tf.get_variable(
            'biases_{}'.format(n_layer + 1),
            shape=[dense_output_dim],
            initializer=tf.zeros_initializer(),
        )
        a = ACTIVATIONS[activation](a @ weights + biases)
        a = tf.nn.dropout(a, keep_prob=(1 - dense_dropout_place))
        dense_input_dim = dense_output_dim

    # output layer
    weights_output = tf.get_variable(
        'weights_output',
        shape=[dense_output_dim, output_dim],
        initializer=tf.keras.initializers.lecun_uniform(seed=seed_base + 3),
    )
    biases_output = tf.get_variable(
        'biases_output',
        shape=[output_dim],
        initializer=tf.zeros_initializer(),
    )
    # l2_loss_output = tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(biases_output)
    output_before_softmax = a @ weights_output + biases_output
    output_ = tf.nn.softmax(output_before_softmax)
    output = tf.identity(output_, name=OP_INFERENCE)

    # get loss
    loss_ = tf.losses.softmax_cross_entropy(
        y_place,
        output_before_softmax,
        weights=sample_weight_place,
        label_smoothing=0.0,
    )
    loss = tf.reduce_mean(
        loss_,
        name=OP_LOSS,
    )
    # loss = loss + l2_regularize * (l2_loss_dense + l2_loss_output)

    # training
    tf.train.GradientDescentOptimizer(lr_place).minimize(
        loss,
        name=OP_TRAIN,
    )

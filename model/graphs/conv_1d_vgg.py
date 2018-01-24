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


def build_conv_1d_vgg(
        input_dim,
        output_dim,
        conv_structure,
        dense_structure,
        samplewise_norm=False,
        batch_norm=True,
        do_global_pooling=True,
        log_epsilon=None,
        conv_with_bias=False,
        seed_base=2017,
    ):
    x_place, y_place, sample_weight_place, lr_place, _, conv_dropout_place, dense_dropout_place, is_training = get_input(input_dim, output_dim)
    if samplewise_norm:
        # TODO: normalize sample-wise
        mean = tf.reduce_mean(x_place, axis=1)
        std = tf.keras.backend.std(x_place, axis=1)
        print(mean.shape)
        print(std.shape)
        x_place = (tf.transpose(x_place) - mean) / std
        x_place = tf.transpose(x_place)
    if log_epsilon is not None:
        x_place = tf.log(tf.abs(x_place) + log_epsilon)
    x_place_reshape = tf.expand_dims(x_place, axis=1)
    print(x_place_reshape.shape)

    conv_out = x_place_reshape
    conv_outs = []
    n_input_channel = conv_out.shape[1]
    for idx_conv, (
            n_kernels,
            kernel_width,
            kernel_stride,
            activation,
        ) in enumerate(conv_structure):
        if batch_norm:
            conv_out = tf.layers.batch_normalization(
                conv_out,
                axis=1,
                training=is_training,
                fused=True,
            )

        if activation == 'pooling':
            conv_out = tf.layers.average_pooling1d(
                tf.transpose(conv_out, perm=[0, 2, 1]),
                pool_size=(kernel_width,),
                strides=(kernel_stride,),
                data_format='channels_first',
                name="conv_1d_pool_{}".format(idx_conv),
            )
            conv_out = tf.transpose(conv_out, [0, 2, 1])
            conv_out = tf.nn.dropout(conv_out, keep_prob=(1 - conv_dropout_place))
        elif activation == 'skip':
            conv_outs.append(tf.layers.flatten(tf.transpose(tf.layers.average_pooling1d(
                tf.transpose(conv_out, perm=[0, 2, 1]),
                pool_size=(conv_out.shape[2],),
                strides=(conv_out.shape[2],),
                data_format='channels_first',
                name="global_pool",
            ), perm=[0, 2, 1])))
            print(conv_outs)
        else:
            kernel = tf.get_variable(
                'kernel_weights_{}'.format(idx_conv+1),
                shape=[kernel_width, n_input_channel, n_kernels],
                initializer=tf.keras.initializers.lecun_uniform(seed=seed_base + idx_conv),
            )
            conv_out = tf.nn.convolution(
                input=conv_out,
                filter=kernel,
                padding='SAME',
                strides=(kernel_stride,),
                dilation_rate=None,
                name='conv_{}'.format(idx_conv),
                data_format='NCW'
            )
            if conv_with_bias:
                bias = tf.get_variable(
                    'kernel_bias_{}'.format(idx_conv+1),
                    shape=conv_out.shape[1:2],
                    initializer=tf.zeros_initializer(),
                )
                conv_out = tf.transpose(conv_out, [0, 2, 1]) + bias
                conv_out = tf.transpose(conv_out, [0, 2, 1])
            conv_out = ACTIVATIONS[activation](conv_out)
            n_input_channel = n_kernels
        print(conv_out.shape)

    if do_global_pooling:
        conv_outs.append(tf.layers.flatten(tf.transpose(tf.layers.average_pooling1d(
            tf.transpose(conv_out, perm=[0, 2, 1]),
            pool_size=(conv_out.shape[2],),
            strides=(conv_out.shape[2],),
            data_format='channels_first',
            name="global_pool",
        ), perm=[0, 2, 1])))
    else:
        conv_outs.append(tf.layers.flatten(conv_out))
    print(conv_outs)

    dense_input = tf.concat(conv_outs, axis=1)
    print(dense_input.shape)
    a = dense_input
    dense_input_dim = a.shape[1]
    dense_output_dim = dense_input_dim
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

import tensorflow as tf

from ..names import (
    WAVELET_DROPOUT_PLACE,
    CONV_DROPOUT_PLACE,
    DENSE_DROPOUT_PLACE,
    OP_INFERENCE,
    OP_LOSS,
    OP_TRAIN,
)
from .common import (
    ACTIVATIONS,
    get_input,
)


def build_wavelet_1d_2d_cnn_mlp(
        input_dim,
        output_dim,
        should_share_wavelet,
        n_wavelets,
        wavelet_length,
        wavelet_range,
        conv_structure,
        dense_structure,
        l2_regularize,
        seed_base=2017,
    ):
    x_place, y_place, sample_weight_place, lr_place = get_input(input_dim, output_dim)
    wavelet_dropout_place = tf.placeholder_with_default(
        0.0,
        shape=(),
        name=WAVELET_DROPOUT_PLACE,
    )
    conv_dropout_place = tf.placeholder_with_default(
        0.0,
        shape=(),
        name=CONV_DROPOUT_PLACE,
    )
    dense_dropout_place = tf.placeholder_with_default(
        0.0,
        shape=(),
        name=DENSE_DROPOUT_PLACE,
    )

    # wavelet layers
    x_place_reshape = tf.expand_dims(x_place, axis=1)
    print(x_place_reshape.shape)
    if should_share_wavelet:
        wavelets = tf.get_variable(
            'wavelet_weights',
            shape=[wavelet_length, 1, n_wavelets],  # [wavelet_size, n_channel, n_wavelet]
            initializer=tf.keras.initializers.lecun_uniform(seed=seed_base - 1),
        )
    imfs = []
    for k in wavelet_range:
        if not should_share_wavelet:
            wavelets = tf.get_variable(
                'wavelet_weights_{}'.format(k),
                shape=[k, 1, n_wavelets],  # [wavelet_size, n_channel, n_wavelet]
                initializer=tf.keras.initializers.lecun_uniform(seed=seed_base - 1),
            )
            imf = tf.nn.convolution(
                input=x_place_reshape,
                filter=wavelets,
                padding='SAME',
                strides=None,
                dilation_rate=None,
                name="wavelet_1d_conv_{}".format(k),
                data_format='NCW'
            )
        else:
            imf = tf.nn.convolution(
                input=x_place_reshape,
                filter=wavelets,
                padding='SAME',
                strides=None,
                dilation_rate=(k,),
                name="wavelet_1d_conv_{}".format(k),
                data_format='NCW'
            )
        pooled_imf = tf.layers.average_pooling1d(
            tf.transpose(imf, perm=[0, 2, 1]),
            pool_size=(4,),
            strides=(4,),
            data_format='channels_first',
            name="wavelet_1d_pool_{}".format(k),
        )
        imfs.append(pooled_imf)
    wavelet_out = tf.stack(imfs, axis=1)
    wavelet_out = tf.nn.tanh(wavelet_out)
    wavelet_out = tf.nn.dropout(wavelet_out, keep_prob=(1 - wavelet_dropout_place))
    print(imf.shape)
    print(pooled_imf.shape)
    print(wavelet_out.shape)

    # conv layers
    conv_out = wavelet_out
    n_input_channel = n_wavelets
    for n_layer, (w, h, sw, sh, n_kernel, activation) in enumerate(conv_structure):
        if activation == 'pooling':
            conv_out = tf.layers.max_pooling2d(
                conv_out,
                pool_size=(w, h),
                strides=(sw, sh),
                padding='valid',
                data_format='channels_last',
                name=None
            )
            conv_out = tf.nn.dropout(conv_out, keep_prob=(1 - conv_dropout_place))
            print(conv_out.shape)
        else:
            kernel = tf.get_variable(
                'kernel_weights_{}'.format(n_layer+1),
                shape=[w, h, n_input_channel, n_kernel],
                initializer=tf.keras.initializers.lecun_uniform(seed=seed_base + n_layer),
            )
            conv_out = tf.nn.convolution(
                input=conv_out,
                filter=kernel,
                padding='SAME',
                strides=(sw, sh),
                dilation_rate=None,
                name='conv1',
                data_format='NHWC'
            )
            print(conv_out.shape)
            conv_out = ACTIVATIONS[activation](conv_out)
            n_input_channel = n_kernel

        print(conv_out.shape)

    # Dense Layer
    a = tf.layers.flatten(conv_out)
    print(a.shape)
    dense_input_dim = a.shape[1]
    l2_loss_dense = 0
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
        l2_loss_dense += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
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
    l2_loss_output = tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(biases_output)
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
    loss = loss + l2_regularize * (l2_loss_dense + l2_loss_output)

    # training
    tf.train.AdadeltaOptimizer(lr_place).minimize(
        loss,
        name=OP_TRAIN,
    )

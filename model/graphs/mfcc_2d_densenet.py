import tensorflow as tf

from ..names import (
    OP_INFERENCE,
    OP_LOSS,
    OP_TRAIN,
)
from .common import (
    ACTIVATIONS,
    get_input,
    dense_2d_block,
)


def build_mfcc_2d_densenet(
        input_dim,
        output_dim,
        frame_length,
        frame_step,
        n_mfccs,
        num_mel_bins,
        densenet_structure,
        dense_structure,
        do_global_pooling=False,
        seed_base=2017,
    ):
    x_place, y_place, sample_weight_place, lr_place, _, conv_dropout_place, dense_dropout_place, is_training = get_input(input_dim, output_dim)

    stfts = tf.contrib.signal.stft(
        x_place,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=None,
    )
    magnitude_spectrograms = tf.abs(stfts)
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 40.0, 7600.0
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        16000,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms,
        linear_to_mel_weight_matrix,
        1,
    )
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]
        )
    )
    log_offset = 1e-8
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
    # Keep the first `num_mfccs` MFCCs.
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms,
    )[..., :n_mfccs]
    print(mfccs.shape)

    # conv layers
    n_input_channel = 1
    mfccs = tf.expand_dims(mfccs, axis=1)

    dense_conv_output = mfccs
    for idx_dense_block, (
            n_layers,
            n_kernels,
            n_compressed_kernels,
            kernel_height,
            kernel_width,
            activation,
            pool_height,
            pool_width,
        ) in enumerate(densenet_structure):
        output_wave = dense_2d_block(
            input_wave=dense_conv_output,
            is_training=is_training,
            n_layers=n_layers,
            n_kernels=n_kernels,
            with_compressed=(n_compressed_kernels is not None),
            n_compressed_kernels=n_compressed_kernels,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            activation=activation,
            name='{}_dense_2d_net'.format(idx_dense_block),
        )
        # print(output_wave.shape)
        output_wave_pooled = tf.layers.max_pooling2d(
            output_wave,
            pool_size=(pool_height, pool_width),
            strides=(pool_height, pool_width),
            padding='valid',
            data_format='channels_first',
            name=None
        )
        # print(output_wave_pooled.shape)
        output_wave_pooled = tf.nn.dropout(output_wave_pooled, keep_prob=(1 - conv_dropout_place))
        # print(output_wave_pooled.shape)
        dense_conv_output = tf.nn.dropout(
            output_wave_pooled,
            keep_prob=(1 - conv_dropout_place),
        )
    if do_global_pooling:
        dense_conv_output = tf.layers.max_pooling2d(
            dense_conv_output,
            pool_size=dense_conv_output.shape[2:],
            strides=dense_conv_output.shape[2:],
            data_format='channels_first',
        )

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

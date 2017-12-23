from .graphs import build_wavelet_1d_2d_cnn_mlp


ALL_MODELS = {
    '1': (
        lambda input_dim, output_dim: build_wavelet_1d_2d_cnn_mlp(
            input_dim,
            output_dim,
            should_share_wavelet=True,
            n_wavelets=32,
            wavelet_range=[k for k in range(1, 33)],
            wavelet_length=8,
            conv_structure=[
                (2, 4, 1, 2, 32, 'selu'),
                (2, 4, 1, 2, 32, 'selu'),
                (2, 4, 1, 2, 64, 'selu'),
                # (1, 2, 1, 2, -1, 'pooling'),
                (2, 4, 1, 2, 64, 'selu'),
                (2, 4, 1, 2, 64, 'selu'),
                (2, 8, 1, 2, 128, 'selu'),
                # (1, 2, 1, 2, -1, 'pooling'),
                (4, 16, 1, 2, 128, 'selu'),
                (4, 16, 1, 2, 256, 'selu'),
                (1, 2, 1, 2, -1, 'pooling'),
            ],
            dense_structure=[
                (2048, 'selu'),
                (2048, 'selu'),
            ],
            l2_regularize=0.00001,
        ), {
            'wavelet_dropout_prob': 0.1,
            'conv_dropout_prob': 0.1,
            'dense_dropout_prob': 0.2,
        },
    ),
    '2': (
        lambda input_dim, output_dim: build_wavelet_1d_2d_cnn_mlp(
            input_dim,
            output_dim,
            n_wavelets=32,
            wavelet_range=[4*k for k in range(1, 25)],
            wavelet_length=-1,
            conv_structure=[
                (2, 4, 1, 2, 32, 'selu'),
                (2, 4, 1, 2, 32, 'selu'),
                (2, 4, 1, 2, 64, 'selu'),
                # (1, 2, 1, 2, -1, 'pooling'),
                (2, 4, 1, 2, 64, 'selu'),
                (2, 4, 1, 2, 64, 'selu'),
                (2, 8, 1, 2, 128, 'selu'),
                # (1, 2, 1, 2, -1, 'pooling'),
                (4, 16, 1, 2, 128, 'selu'),
                (4, 16, 1, 2, 256, 'selu'),
                (1, 2, 1, 2, -1, 'pooling'),
            ],
            dense_structure=[
                (2048, 'selu'),
                (2048, 'selu'),
            ],
            should_share_wavelet=False,
            l2_regularize=0.00001,
        ), {
            'wavelet_dropout_prob': 0.0,
            'conv_dropout_prob': 0.0,
            'dense_dropout_prob': 0.1,
        },
    ),
    '3': (
        lambda input_dim, output_dim: build_wavelet_1d_2d_cnn_mlp(
            input_dim,
            output_dim,
            n_wavelets=32,
            wavelet_range=[4*k for k in range(1, 11)],
            wavelet_length=-1,
            conv_structure=[
                (2, 4, 1, 2, 32, 'selu'),
                (2, 4, 1, 2, 32, 'selu'),
                (1, 2, 1, 2, -1, 'pooling'),
                (2, 4, 1, 2, 32, 'selu'),
                (2, 8, 1, 2, 64, 'selu'),
                (1, 2, 1, 2, -1, 'pooling'),
                (4, 16, 1, 2, 64, 'selu'),
                (2, 4, 1, 2, -1, 'pooling'),
            ],
            dense_structure=[
                (1024, 'selu'),
                (512, 'selu'),
            ],
            should_share_wavelet=False,
            l2_regularize=0.00001,
        ), {
            'wavelet_dropout_prob': 0.2,
            'conv_dropout_prob': 0.2,
            'dense_dropout_prob': 0.5,
        },
    ),
    '4': (
        lambda input_dim, output_dim: build_wavelet_1d_2d_cnn_mlp(
            input_dim,
            output_dim,
            n_wavelets=16,
            wavelet_range=[4*k for k in range(1, 9)],
            wavelet_length=-1,
            conv_structure=[
                (2, 4, 1, 2, 32, 'selu'),
                (1, 2, 1, 2, -1, 'pooling'),
                (2, 4, 1, 2, 32, 'selu'),
                (1, 2, 1, 2, -1, 'pooling'),
                (4, 16, 1, 2, 64, 'selu'),
                (2, 4, 1, 2, -1, 'pooling'),
            ],
            dense_structure=[
                (512, 'selu'),
            ],
            should_share_wavelet=False,
            l2_regularize=0.00001,
        ), {
            'wavelet_dropout_prob': 0.0,
            'conv_dropout_prob': 0.0,
            'dense_dropout_prob': 0.0,
        },
    ),

}

"""
"""

from typing import Tuple, Union, Optional, Callable
from collections.abc import Iterable

import tensorflow as tf


class StereoNet(tf.keras.models.Model):
    """
    """

    def __init__(self, k: int = 3, r: int = 3, max_disp: int = 192):
        super().__init__(name='')

        self.k = k
        self.r = r
        self.max_disp = max_disp
        self.disp = (self.max_disp + 1) // (2**self.k)

        self.feature_extraction = get_downsampling_feature_network(k=self.k)
        self.cost_volume = get_cost_volume_network(k=4)
        # self.disparity_regression = get_disparity_regression_network(self.disp)
        self.edge_aware_refinement = EdgeAwareRefinement(k=4)

    def call(self, left, right, training=False):

        reference_embedding = self.feature_extraction(left, training=training)
        target_embedding = self.feature_extraction(right, training=training)

        cost = tf.TensorArray(dtype=tf.float32, size=self.disp)
        cost.write(0, reference_embedding - target_embedding)
        for idx in range(1, self.disp):
            cost.write(idx, tf.pad((reference_embedding[:, :, idx:, :] - target_embedding[:, :, :-idx, :]), [[0, 0], [0, 0], [idx, 0], [0, 0]]))
        cost = cost.stack()
        cost = tf.transpose(cost, perm=[1, 0, 2, 3, 4])

        cost = self.cost_volume(cost, training=training)
        cost = tf.squeeze(cost, axis=-1)

        pred = tf.keras.layers.Softmax(axis=1)(cost)

        # Disparity regression
        disp = tf.reshape(tf.range(self.disp, dtype=tf.float32), (1, self.disp, 1, 1))
        disp = tf.repeat(tf.repeat(tf.repeat(disp, pred.shape[0], axis=0), pred.shape[2], axis=2), pred.shape[3], axis=3)  # This feels very inelegant
        pred = tf.reduce_sum(tf.math.multiply(disp, pred), axis=1)

        pred_bottom = pred * left.shape[2] / pred.shape[2]
        pred_bottom = tf.image.resize(pred_bottom[..., tf.newaxis], size=left.shape[1:3])
        pred_bottom = tf.squeeze(pred_bottom, axis=-1)

        pred_top = self.edge_aware_refinement(pred, left)

        pred_pyramid = tf.TensorArray(tf.float32, size=2)
        pred_pyramid.write(0, pred_bottom)
        pred_pyramid.write(1, pred_top)

        pred_pyramid = pred_pyramid.stack()
        pred_pyramid = tf.transpose(pred_pyramid, perm=[1, 2, 3, 0])

        return pred_pyramid


class ResBlock(tf.keras.models.Model):
    """ https://www.tensorflow.org/tutorials/customization/custom_layers#models_composing_layers
    """

    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]],
                 padding: Union[str, Tuple[int, int], int],
                 dilation_rate: Union[int, Tuple[int, int]],
                 downsampler: Optional[Callable]):
        super().__init__(name='')

        if isinstance(padding, int) or isinstance(padding, Iterable):
            self.conv = tf.keras.models.Sequential()
            self.conv.add(tf.keras.layers.ZeroPadding2D(padding=padding))
            self.conv.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate))
        if isinstance(padding, str):
            self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU(negative_slope=0.2)

        self.downsampler = downsampler

    def call(self, input_tensor, training=False):
        x = input_tensor

        x = self.conv(x)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)

        if self.downsampler is not None:
            input_tensor = self.downsampler(input_tensor)

        return x + input_tensor


def get_downsampling_feature_network(k: int = 3) -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, None, 3)))

    for _ in range(k):
        model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='valid'))

    for _ in range(6):
        model.add(ResBlock(filters=32, kernel_size=(3, 3), strides=(1, 1), padding=1, dilation_rate=(1, 1), downsampler=None))

    model.add(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1)))

    return model


def get_cost_volume_network(k: int = 4) -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, None, None, 32)))

    for _ in range(k):
        model.add(tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)))
        model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU(negative_slope=0.2))

    model.add(tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)))
    model.add(tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1)))

    return model


class EdgeAwareRefinement(tf.keras.models.Model):
    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
        self.atrous_list = [1, 2, 4, 8, 1, 1]

        self.feature_conv = tf.keras.models.Sequential([tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
                                                        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1)),
                                                        tf.keras.layers.ReLU(negative_slope=0.2)
                                                        ])

        self.residual_atrous_net = tf.keras.models.Sequential()
        for dilation_rate in self.atrous_list:
            padding = 1
            if dilation_rate > padding:
                padding = dilation_rate
            self.residual_atrous_net.add(ResBlock(filters=32, kernel_size=3, strides=(1, 1), padding=padding, dilation_rate=dilation_rate, downsampler=None))

        self.final_conv = tf.keras.models.Sequential([tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
                                                      tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1)),
                                                      ])

    def call(self, disp, colour):
        _, _, original_disp_h = tf.shape(disp)
        disp = tf.image.resize(disp[..., tf.newaxis], size=tf.shape(colour)[1:3])

        if tf.shape(colour)[2] / original_disp_h >= 1.5:
            disp *= 8

        output = tf.concat([disp, colour], axis=-1)
        output = self.feature_conv(output)
        output = self.residual_atrous_net(output)

        output = self.final_conv(output)
        output += disp

        output = tf.keras.activations.relu(output)
        output = tf.squeeze(output, axis=-1)
        
        return output


def main():
    import numpy as np

    stereo_net = StereoNet(k=3, r=3, max_disp=192)
    out = stereo_net(left=np.random.random((1, 540, 960, 3)), right=np.random.random((1, 540, 960, 3)))

    print('stall')


if __name__ == "__main__":
    main()

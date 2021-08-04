"""
"""

from typing import Tuple

from pathlib import Path

import tensorflow as tf


def read_PFM(path: Path) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Tensorflow reimplementation of https://github.com/zhixuanli/StereoNet/blob/f5576689e66e8370b78d9646c00b7e7772db0394/dataloader/readpfm.py
    """

    channels = None
    width = None
    height = None
    scale = None
    endian = None

    raw = tf.io.read_file(str(path))

    gatherer = ''
    breakline = 0
    for idx, char in enumerate(tf.strings.bytes_split(raw)):
        if breakline > 2:
            break

        if char == '\n':

            if breakline == 0:
                gatherer = tf.strings.strip(gatherer)
                if gatherer == b'PF':
                    channels = 3
                elif gatherer == b'Pf':
                    channels = 1
                else:
                    raise Exception('Not a PFM file.')

            if breakline == 1:
                gatherer = tf.strings.strip(gatherer)
                dim_match = tf.strings.regex_full_match(gatherer, r'^(\d+)\s(\d+)$')
                if dim_match:
                    width, height = tf.strings.split(gatherer, ' ')
                    width, height = tf.strings.to_number(width, tf.dtypes.int32), tf.strings.to_number(height, tf.dtypes.int32)
                    shape = (height, width, channels)
                else:
                    raise Exception('Malformed PFM header.')

            if breakline == 2:
                scale = tf.strings.to_number(gatherer, tf.dtypes.float32)
                endian = tf.cond(scale < 0, lambda: '<', lambda: '>')
                scale = tf.cond(scale < 0, lambda: -scale, lambda: scale)

                if endian == '>':
                    raise Exception("Big endian hasn't been implemented.")
            breakline += 1
            gatherer = ''

            continue

        gatherer = tf.strings.join([gatherer, char])

    raw = tf.strings.substr(raw, idx, -1)  # Negative value means take the rest
    img = tf.io.decode_raw(raw, tf.float32)
    img = tf.reshape(img, shape)
    img = tf.image.flip_up_down(img)

    return img, scale


def main():
    read_PFM(path=Path(r"C:\Users\andre\Documents\Python\StereoNet_TF2\data\SceneFlow\disparity\TRAIN\A\0000\left\0006.pfm"))


if __name__ == "__main__":
    main()

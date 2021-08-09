"""
"""

from typing import Optional, Tuple

import argparse
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt


def parent_parser():
    parser = argparse.ArgumentParser(description='Arguments to train/test the StereoNet model')
    return parser


def dataset_args(parser: argparse.ArgumentParser) -> None:
    dataset_group = parser.add_argument_group('dataset', description='Dataset related arguments')
    dataset_group.add_argument('--data_root_path', type=str)
    dataset_group.add_argument('--batch_size', type=int, default=32)
    dataset_group.add_argument('--crop_size', type=int, nargs=2, default=[256,513])
    dataset_group.add_argument('--shuffle_buffer', type=int, default=1000)
    

def model_args(parser: argparse.ArgumentParser) -> None:
    model_group = parser.add_argument_group('model', description='Model related arguments')
    model_group.add_argument('--candidate_disparities', type=int, default=192)
    model_group.add_argument('--k', type=int, default=4)


def training_args(parser: argparse.ArgumentParser) -> None:
    training_group = parser.add_argument_group('training', description='Training related arguments')
    training_group.add_argument('--checkpoint_path', type=str)
    training_group.add_argument('--epochs', type=int, default=20)
    training_group.add_argument('--learning_rate', type=float, default=1e-3)
    training_group.add_argument('--max_disp', type=int, default=160)


def parse_training_args():
    """
    """

    # Dataset related arguments
    parser = parent_parser()
    model_args(parser)
    training_args(parser)
    dataset_args(parser)

    config = parser.parse_args()

    return config


def read_image(filename: tf.Tensor) -> tf.Tensor:
    """
    Tensorflow really complains if I try and allow filename to be a Path object
    """
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    return image


def read_PFM(path: Path) -> tf.Tensor:
    """
    Tensorflow reimplementation of https://github.com/zhixuanli/StereoNet/blob/f5576689e66e8370b78d9646c00b7e7772db0394/dataloader/readpfm.py
    """

    channels = None
    width = None
    height = None
    scale = None
    little_endian = None

    raw = tf.io.read_file(str(path))
    raw = tf.strings.split(raw, '\n')

    channels = tf.cond(tf.strings.strip(raw[0]) == b'Pf', lambda: 1, lambda: 3)
    shape_string = tf.strings.strip(raw[1])
    dim_match = tf.strings.regex_full_match(shape_string, r'^(\d+)\s(\d+)$')
    if not dim_match:
        raise Exception("Malformed PFM header.")
    width, height = tf.strings.split(shape_string, ' ')
    width, height = tf.strings.to_number(width, tf.dtypes.int32), tf.strings.to_number(height, tf.dtypes.int32)

    scale = tf.strings.to_number(raw[2], tf.float32)
    little_endian = tf.cond(scale < 0, lambda: True, lambda: False)
    scale = tf.math.abs(scale)

    img = tf.strings.join(raw[3:], '\n')
    img = tf.io.decode_raw(img, tf.float32, little_endian=little_endian)
    img = tf.reshape(img, (height, width, channels))
    img = tf.image.flip_up_down(img)

    return img


def read_PFM_tensor(tensor_path: tf.Tensor) -> tf.Tensor:
    """
    Function which can be mapped over a Tensorflow dataset to taking in string paths and return tensors.
    """
    return tf.io.parse_tensor(tf.io.read_file(tensor_path), tf.float32)


def write_pfm_as_tensor(path: Path, new_extension: str = '.tensor') -> None:
    """
    Takes in the path to a PFM file, reads the PFM file, then writes out a tensor.
    """
    output_path = path.with_suffix(new_extension)

    pfm = read_PFM(path)

    tf.io.write_file(str(output_path), tf.io.serialize_tensor(pfm))


def plot_batch(batch: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor], show: bool = False, max_pairs: int = 4, ax: Optional[plt.Axes] = None):
    """
    Helper function to plot a few examples from a normalized batch
    """
    (left, right), disp = batch
    
    if left.ndim not in [3, 4]:
        raise Exception("Malformed input shape.")

    if left.ndim == 3:
        left = tf.expand_dims(left, 0)
        right = tf.expand_dims(right, 0)
        disp = tf.expand_dims(disp, 0)


    max_pairs = min(left.shape[0], max_pairs) 

    if not ax:
        fig, ax = plt.subplots(ncols=3, nrows=max_pairs)

    left = tf.cast((left + 1) * 127.5, tf.uint8)
    right = tf.cast((right + 1) * 127.5, tf.uint8)

    for idx, (l, r, d) in enumerate(zip(left, right, disp)):
        if idx >= max_pairs:
            break
        ax[idx][0].imshow(l)
        ax[idx][1].imshow(r)
        ax[idx][2].imshow(d)

    plt.tight_layout()
    
    if not show:
        return ax
    
    plt.show()


def convert_pfm_to_tensor(path: Path):
    """
    I couldn't find an elegant way to create a Tensorflow dataset object with PFM files.
    Faster to convert the pfm's to tensors.

    This function will convert each pfm in a nested directory to a tensor that can easily
    be loaded into a dataset object for training.
    """
    for disp_path in tqdm(path.rglob('*.pfm')):
        if disp_path.with_suffix('.tensor').exists():
            if disp_path.stat().st_size < 100000:
                raise Exception(f"Something is probably wrong with this file: {disp_path}.")
            continue
        write_pfm_as_tensor(disp_path, new_extension='.tensor')


def main():
    path = Path(r'C:\Users\andre\Documents\Python\StereoNet_TF2\data\SceneFlow')
    convert_pfm_to_tensor(path)


if __name__ == "__main__":
    main()

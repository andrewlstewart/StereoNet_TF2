"""
"""

from typing import Optional
from pathlib import Path
import random
import argparse

import tensorflow as tf

import src.utils as utils


def make_dataset(sceneflow_path: str,
                 string_exclude: Optional[str] = None,
                 string_include: Optional[str] = None,
                 shuffle: bool = False) -> tf.data.Dataset:
    """ 
    string_exclude: If this string appears in the parent path of an image, don't add them to the dataset
    string_include: If this string DOES NOT appear in the parent path of an image, don't add them to the dataset
    """

    left_image_path = []
    right_image_path = []
    left_disp_path = []

    # For each left image, do some path manipulation to find the corresponding right
    # image and left disparity.
    sceneflow_path = Path(sceneflow_path)
    for path in sceneflow_path.rglob('*.png'):
        if not 'left' in path.parts:
            continue

        if string_exclude and string_exclude in path.parts:
            continue
        if string_include and string_include not in path.parts:
            continue

        r_path = Path("\\".join(['right' if 'left' in part else part for part in path.parts]))
        d_path = Path("\\".join([f'{part.replace("frames_cleanpass","")}disparity' if 'frames_cleanpass' in part else part for part in path.parts])).with_suffix('.tensor')
        # assert r_path.exists()
        # assert d_path.exists()

        if not r_path.exists() or not d_path.exists():
            continue

        left_image_path.append(path)
        right_image_path.append(r_path)
        left_disp_path.append(d_path)

    # Tensorflow hates Path objects
    left_image_filenames = [str(p) for p in left_image_path]
    right_image_filenames = [str(p) for p in right_image_path]
    left_disp_filenames = [str(p) for p in left_disp_path]

    if shuffle:
        random_list = [random.random() for _ in left_image_filenames]

        def shuffler(random_list):
            for random in random_list:
                yield random
        # Because the following shufflers are all using the same random_list, they will yield
        # the same values.  Use this as the key to 'sort' (ie. shuffle).
        # https://stackoverflow.com/a/67598880
        left_shuffler = shuffler(random_list)
        right_shuffler = shuffler(random_list)
        disp_shuffler = shuffler(random_list)

        left_image_filenames = sorted(left_image_filenames, key=lambda _: next(left_shuffler))
        right_image_filenames = sorted(right_image_filenames, key=lambda _: next(right_shuffler))
        left_disp_filenames = sorted(left_disp_filenames, key=lambda _: next(disp_shuffler))

    left_image_filenames_ds = tf.data.Dataset.from_tensor_slices(left_image_filenames)
    right_image_filenames_ds = tf.data.Dataset.from_tensor_slices(right_image_filenames)
    left_disp_filenames_ds = tf.data.Dataset.from_tensor_slices(left_disp_filenames)

    left_image_ds = left_image_filenames_ds.map(utils.read_image, num_parallel_calls=tf.data.AUTOTUNE)
    right_image_ds = right_image_filenames_ds.map(utils.read_image, num_parallel_calls=tf.data.AUTOTUNE)
    left_disp_ds = left_disp_filenames_ds.map(utils.read_PFM_tensor, num_parallel_calls=tf.data.AUTOTUNE)

    image_ds = tf.data.Dataset.zip((left_image_ds, right_image_ds))
    ds = tf.data.Dataset.zip((image_ds, left_disp_ds))

    return ds


def rescale(image: tf.Tensor):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def augmentations(batch: tf.Tensor, seed: tf.Tensor, config: argparse.Namespace) -> tf.Tensor:
    """
    Currently only performing the same random crop on the left, right, and disparity maps
    """
    (left, right), disp = batch

    # Randomly crop.  Passing the same seed to all crop functions 'should' get the same cropped regions
    # left = tf.image.stateless_random_crop(left, size=[left.shape[0], spatial_crop_shape[0], spatial_crop_shape[1], left.shape[3]], seed=seed)
    # right = tf.image.stateless_random_crop(right, size=[right.shape[0], spatial_crop_shape[0], spatial_crop_shape[1], right.shape[3]], seed=seed)
    # disp = tf.image.stateless_random_crop(disp, size=[disp.shape[0], spatial_crop_shape[0], spatial_crop_shape[1], disp.shape[3]], seed=seed)

    left = tf.image.stateless_random_crop(left, size=(config.batch_size, config.crop_size[0], config.crop_size[1], 3), seed=seed)
    right = tf.image.stateless_random_crop(right, size=(config.batch_size, config.crop_size[0], config.crop_size[1], 3), seed=seed)
    disp = tf.image.stateless_random_crop(disp, size=(config.batch_size, config.crop_size[0], config.crop_size[1], 1), seed=seed)

    return (left, right), disp


def dataset_prepare(ds: tf.data.Dataset,
                    config: Optional[argparse.Namespace] = None,
                    apply_augmentations: bool = False) -> tf.data.Dataset:
    """
    https://www.tensorflow.org/tutorials/images/data_augmentation#apply_the_preprocessing_layers_to_the_datasets
    """

    ds = ds.map(lambda images, disp: ((rescale(images[0]), rescale(images[1])), disp))

    if config.shuffle_buffer:
        ds = ds.shuffle(config.shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.batch(config.batch_size)

    if apply_augmentations:
        counter = tf.data.experimental.Counter()
        ds = tf.data.Dataset.zip((ds, (counter, counter)))
        ds = ds.map(lambda inst, seed: augmentations(inst, seed, config), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def main():

    config = utils.parse_args()

    train_ds = make_dataset(config.data_root_path, string_exclude='TEST', shuffle=True)
    val_ds = make_dataset(config.data_root_path, string_include='TEST')

    train_ds = dataset_prepare(train_ds, config, apply_augmentations=True)
    val_ds = dataset_prepare(val_ds, config)

    for b in train_ds:
        utils.plot_batch(b, show=True)
        break


if __name__ == "__main__":
    main()

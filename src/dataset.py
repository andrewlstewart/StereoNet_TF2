"""
"""

from pathlib import Path
from glob import glob

import tensorflow as tf

import utils


def parse_image(filename: Path) -> None:
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    return image

def parse_disp(filename: Path) -> None:
    return utils.read_PFM(filename)


def make_dataset(sceneflow_path: Path, batch_size: int) -> tf.data.Dataset:
    
    # ========================= flying =======================
    path = sceneflow_path / 'frames_cleanpass' / 'TRAIN'

    left_image_filenames = list(path.glob('*/*/left/*.png'))
    right_image_filenames = [p.parents[1] / 'right' / p.name for p in left_image_filenames]
    left_disp_filenames = [(p.parents[5] / 'disparity' / p.relative_to(p.parents[4])).with_suffix('.pfm') for p in left_image_filenames]

    left_image_filenames = [str(p) for p in left_image_filenames]
    right_image_filenames = [str(p) for p in right_image_filenames]
    left_disp_filenames = [str(p) for p in left_disp_filenames]

    left_image_filenames_ds = tf.data.Dataset.from_tensor_slices(left_image_filenames)
    right_image_filenames_ds = tf.data.Dataset.from_tensor_slices(right_image_filenames)
    left_disp_filenames_ds = tf.data.Dataset.from_tensor_slices(left_disp_filenames)
    
    left_image_ds = left_image_filenames_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    right_image_ds = right_image_filenames_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    left_disp_ds = left_disp_filenames_ds.map(parse_disp, num_parallel_calls=tf.data.AUTOTUNE)
    
    image_ds = tf.data.Dataset.zip((left_image_ds, right_image_ds))
    ds = tf.data.Dataset.zip((image_ds, left_disp_ds))
    
    return ds


path = Path(r'C:\Users\andre\Documents\Python\StereoNet_TF2\data\SceneFlow')
make_dataset(path, 4)

IMG_SIZE = 540

def make_dataset(path, batch_size):

  def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image

  def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  classes = os.listdir(path)
  filenames = glob(path + '/*/*/*')
#   random.shuffle(filenames)
#   labels = [classes.index(name.split('/')[-2]) for name in filenames]

  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
  images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  ds = tf.data.Dataset.zip((images_ds, labels_ds))
  ds = configure_for_performance(ds)

  return ds

make_dataset(r'C:\Users\andre\Documents\Python\StereoNet_TF2\data\SceneFlow\frames_cleanpass\TRAIN\A', 4)

def get_scene_flow(path: Path) -> tf.data:
    # https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#downloads
    # https://github.com/JiaRenChang/PSMNet
    pass


def get_kitti_2015(path: Path) -> tf.data:

    left_img_path = path / 'image_2'
    right_img_path = path / 'image_3'
    left_disp_path = path / 'disp_occ_0'
    right_disp_path = path / 'disp_occ_1'

    for file_path in left_img_path.glob('*'):
        a=1

    datasets = {}
    for name, path in [('left_img', left_img_path), ('right_img', right_img_path), ('left_disp', left_disp_path), ('right_disp', right_disp_path)]:
        datasets[name] = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                                             label_mode=None,
                                                                             shuffle=False,
                                                                             image_size=(375, 1242),
                                                                             batch_size=1)

    print('stall')


def main():
    root_path = Path.cwd() / 'data' / 'Kitti_2015' / 'data_scene_flow' / 'training'
    ds = get_kitti_2015(root_path)


if __name__ == "__main__":
    main()

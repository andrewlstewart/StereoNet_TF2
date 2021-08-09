"""
"""

from pathlib import Path
from datetime import datetime

import tensorflow as tf

import src.model as model
import src.dataset as dataset
import src.utils as utils


def masked_smooth_L1(y_true, y_pred, max_disp):
    mask = y_true < max_disp

    huber = tf.keras.losses.Huber()
    loss = huber(y_true[mask], y_pred[mask])

    return loss


def main():
    config = utils.parse_training_args()

    stereo_net = model.StereoNet(k=config.k, candidate_disparities=config.candidate_disparities)

    train_ds = dataset.make_dataset(config.data_root_path, string_exclude='TEST', shuffle=True)
    val_ds = dataset.make_dataset(config.data_root_path, string_include='TEST')

    train_ds = dataset.dataset_prepare(train_ds, config, apply_augmentations=True)
    val_ds = dataset.dataset_prepare(val_ds, config)

    optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate)

    checkpoint_path = Path(config.checkpoint_path) / datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path))

    tensorboard_logs_path = Path(config.checkpoint_path) / 'logs'
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_logs_path))

    # Haven't investigated but stackoverflow says this is equivalent to SmoothL1Loss
    # https://stackoverflow.com/a/56255595
    def loss(y_true, y_pred): return masked_smooth_L1(y_true, y_pred, config.max_disp)

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    stereo_net.compile(optimizer=optimizer,
                       loss=loss)
    stereo_net.fit(train_ds, epochs=config.epochs, validation_data=val_ds, callbacks=[checkpoint, tensorboard])

    print('stall')


if __name__ == "__main__":
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import time
import csv
import os
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 256
IMG_HEIGHT = 256
CROP_AUG_RATIO = 0.1


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # [-1, 1]
    img = (img - 0.5) / 0.5
    return img


def random_crop(img, output_img_width, output_img_height):
    aug_img_width = int(output_img_width * (1.0 + CROP_AUG_RATIO))
    aug_img_height = int(output_img_height * (1.0 + CROP_AUG_RATIO))
    img = tf.image.resize_with_pad(img,
                                   aug_img_height,
                                   aug_img_width,
                                   method=tf.image.ResizeMethod.BILINEAR)
    # Random crop image.
    img = tf.image.random_crop(img,
                               size=[output_img_height, output_img_width, 3])
    return img


def load_image(image_path, output_img_width, output_img_height):
    # load the raw data from the file as a string
    img = tf.io.read_file(image_path)
    img = decode_img(img)

    img = random_crop(img, output_img_width, output_img_height)
    # img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], method=tf.image.ResizeMethod.BILINEAR)
    # img = tf.image.random_brightness(img, max_delta=0.1)
    # img = tf.clip_by_value(img, 0.0, 1.0)

    # img = tf.image.random_hue(img, max_delta=0.1)
    # img = tf.image.random_saturation(img, lower=0.8, upper=1.2)

    return img


def load_dataset(image_paths,
                 img_width,
                 img_height,
                 max_num_samples,
                 train_split,
                 buffer_size,
                 batch_size,
                 cache=False):
    num_samples = min(max_num_samples, len(image_paths))
    train_range = int(num_samples * train_split)

    train_ds = tf.data.Dataset.from_tensor_slices(image_paths[0:train_range])
    train_ds = train_ds.map(
        lambda img_path: load_image(img_path, img_width, img_height),
        num_parallel_calls=AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        image_paths[train_range:num_samples])
    test_ds = test_ds.map(
        lambda img_path: load_image(img_path, img_width, img_height),
        num_parallel_calls=AUTOTUNE)

    if cache:
        train_ds = train_ds.cache()
        test_ds = test_ds.cache()

    train_ds = train_ds.shuffle(buffer_size).batch(batch_size)
    test_ds = test_ds.shuffle(buffer_size).batch(batch_size)
    return train_ds, test_ds


def get_face_swap_dataset(img_width,
                          img_height,
                          batch_size,
                          buffer_size,
                          max_num_samples,
                          dataset_a_path,
                          dataset_b_path,
                          train_split=0.8,
                          skip_small_images=False,
                          cache=False):
    images_a = []
    images_b = []
    for image in os.listdir(dataset_a_path):
        images_a.append(os.path.join(dataset_a_path, image))
    for image in os.listdir(dataset_b_path):
        images_b.append(os.path.join(dataset_b_path, image))
    print('Dataset A has {} images'.format(len(images_a)))
    print('Dataset B has {} images'.format(len(images_b)))

    a_train_ds, a_test_ds = load_dataset(images_a, img_width, img_height,
                                         max_num_samples, train_split,
                                         buffer_size, batch_size)
    b_train_ds, b_test_ds = load_dataset(images_b, img_width, img_height,
                                         max_num_samples, train_split,
                                         buffer_size, batch_size)

    return a_train_ds, b_train_ds, a_test_ds, b_test_ds

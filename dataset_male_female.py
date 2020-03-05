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


def parse_image_attrs(attrs_file_path):
    with open(attrs_file_path, 'r') as f:
        csv_input = csv.reader(f, delimiter=' ')
        all_attrs = {}
        attr_names = None
        for col in csv_input:
            if attr_names is None:
                attr_names = []
                for attr_name in col:
                    if len(attr_name) != 0:
                        attr_names.append(attr_name)
                continue

            attrs = {}
            image_name = None
            valid_entry_count = 0
            for entry in col:
                if len(entry) != 0:
                    if image_name is None:
                        image_name = entry
                        continue
                    attrs[attr_names[valid_entry_count]] = entry
                    valid_entry_count += 1

            all_attrs[image_name] = attrs
    return all_attrs


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


def get_male_female_dataset(img_width,
                            img_height,
                            batch_size,
                            buffer_size,
                            max_num_samples,
                            dataset_path,
                            train_split=0.8,
                            skip_small_images=False,
                            cache=False):
    male_images = []
    female_images = []
    attrs_file_path = os.path.join(dataset_path, 'list_attr_celeba.txt')
    all_attrs = parse_image_attrs(attrs_file_path)

    start = time.time()
    for image_name in all_attrs:
        example = all_attrs[image_name]
        # Skip bad examples
        if example['Blurry'] == '1' or example['Wearing_Hat'] == '1':
            continue
        # Skip non-exist examples
        image_path = os.path.join(dataset_path, 'img_align_celeba', image_name)
        if not os.path.exists(image_path):
            continue

        # skip small examples
        if skip_small_images:
            im = Image.open(image_path)
            if im.size[0] < IMG_WIDTH or im.size[1] < IMG_HEIGHT:
                continue

        if example['Male'] == '1':
            male_images.append(image_path)
        elif example['Male'] == '-1':
            female_images.append(image_path)

    print('Time taken for metadata is {} sec\n'.format(time.time() - start))
    print('Available male size {}'.format(len(male_images)))
    print('Available female size {}'.format(len(female_images)))

    male_train_ds, male_test_ds = load_dataset(male_images, img_width,
                                               img_height, max_num_samples,
                                               train_split, buffer_size,
                                               batch_size)
    female_train_ds, female_test_ds = load_dataset(female_images, img_width,
                                                   img_height, max_num_samples,
                                                   train_split, buffer_size,
                                                   batch_size)

    return male_train_ds, female_train_ds, male_test_ds, female_test_ds

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import csv
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 256
IMG_HEIGHT = 256

def parse_image_attrs(attrs_file_path):
  with open('/home/fan/dataset/celeb_img_align/list_attr_celeba.txt', 'r') as f:
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
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def load_image(image_path):
  # load the raw data from the file as a string
  img = tf.io.read_file(image_path)
  img = decode_img(img)

  return img

def get_male_female_dataset(batch_size, buffer_size, max_num_samples, dataset_path, cache=False):
    male_images = []
    female_images = []
    attrs_file_path = os.path.join(dataset_path, 'list_attr_celeba.txt')
    all_attrs = parse_image_attrs(attrs_file_path)

    for image_name in all_attrs:
      example = all_attrs[image_name]
      # Skip bad examples
      if example['Blurry'] == '1' or example['Wearing_Hat'] == '1':
          continue
      image_path = os.path.join(dataset_path, 'img_align_celeba', image_name)
      if example['Male'] == '1':
        male_images.append(image_path)
      elif example['Male'] == '-1' and example['Wearing_Lipstick'] == '1':
        female_images.append(image_path)
    
    print('Available male size {}'.format(len(male_images)))
    print('Available female size {}'.format(len(female_images)))

    male_ds = tf.data.Dataset.from_tensor_slices(male_images[0:max_num_samples])
    female_ds = tf.data.Dataset.from_tensor_slices(female_images[0:max_num_samples])

    male_ds = male_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    female_ds = female_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if cache:
        male_ds = male_ds.cache()
        female_ds = female_ds.cache()

    male_ds = male_ds.shuffle(buffer_size).batch(batch_size)
    female_ds = female_ds.shuffle(buffer_size).batch(batch_size)

    return male_ds, female_ds

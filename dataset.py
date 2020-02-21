from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import csv

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # [-1, 1]
  img = (img - 0.5) / 0.5
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(image_name):
  file_path = '/home/fan/dataset/celeb_img_align/img_align_celeba/' + image_name
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)

  return img

def get_male_female_dataset(batch_size, buffer_size, max_num_samples, cache=False):
    male_images = []
    female_images = []
    # 21 for male, 16 for glass
    classifier_index = 21
    
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

    for image_name in all_attrs:
      example = all_attrs[image_name]
      if example['Male'] == '1' and example['Blurry'] == '-1' and example['Wearing_Hat'] == '-1':
        #(example['Goatee'] == '1' or example['Mustache'] == '1' or example['No_Beard'] == '-1' or example['5_o_Clock_Shadow'] == '1'):
        male_images.append(image_name)
      elif example['Male'] == '-1' and example['Blurry'] == '-1' and example['Wearing_Hat'] == '-1' and example['Wearing_Lipstick'] == '1':
        female_images.append(image_name)
    
    print(attr_names)
    print(len(all_attrs))
    print('Available male size {}'.format(len(male_images)))
    print('Available female size {}'.format(len(female_images)))

    print('Male examples: {}'.format(male_images[0:20]))
    print('Female examples: {}'.format(female_images[0:20]))

    male_ds = tf.data.Dataset.from_tensor_slices(male_images[0:max_num_samples])
    female_ds = tf.data.Dataset.from_tensor_slices(female_images[0:max_num_samples])

    male_ds = male_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    female_ds = female_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    if cache:
        male_ds = male_ds.cache()
        female_ds = female_ds.cache()

    male_ds = male_ds.shuffle(buffer_size).batch(batch_size)
    female_ds = female_ds.shuffle(buffer_size).batch(batch_size)

    return male_ds, female_ds


def get_horse_zebra_dataset(batch_size, buffer_size, max_num_samples):
    dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)
    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    train_horses = train_horses.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    train_zebras = train_zebras.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    train_horses = train_horses.take(max_num_samples).cache().shuffle(buffer_size).batch(batch_size)
    train_zebras = train_zebras.take(max_num_samples).cache().shuffle(buffer_size).batch(batch_size)

    test_horses = test_horses.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    test_zebras = test_zebras.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    test_horses = test_horses.take(max_num_samples).cache().shuffle(buffer_size).batch(batch_size)
    test_zebras = test_zebras.take(max_num_samples).cache().shuffle(buffer_size).batch(batch_size)

    return train_horses, train_zebras

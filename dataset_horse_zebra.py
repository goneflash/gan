from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

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
  image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                          preserve_aspect_ratio=False)
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

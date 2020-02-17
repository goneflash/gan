from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

OUTPUT_CHANNELS = 3
BASE_CHANNEL = 8 

def generator():
    return unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

def discriminator():
    return my_discriminator(norm_type='instancenorm', target=False)

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def unet_generator(output_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
  Returns:
    Generator model
  """

  down_stack = [
      pix2pix.downsample(BASE_CHANNEL * 4, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      pix2pix.downsample(BASE_CHANNEL * 8, 4, norm_type),  # (bs, 64, 64, 128)
      pix2pix.downsample(BASE_CHANNEL * 16, 4, norm_type),  # (bs, 32, 32, 256)
      pix2pix.downsample(BASE_CHANNEL * 32, 4, norm_type),  # (bs, 16, 16, 512)
      pix2pix.downsample(BASE_CHANNEL * 32, 4, norm_type),  # (bs, 8, 8, 512)
      pix2pix.downsample(BASE_CHANNEL * 32, 4, norm_type),  # (bs, 4, 4, 512)
      pix2pix.downsample(BASE_CHANNEL * 32, 4, norm_type),  # (bs, 2, 2, 512)
      pix2pix.downsample(BASE_CHANNEL * 32, 4, norm_type),  # (bs, 1, 1, 512)
  ]

  up_stack = [
      pix2pix.upsample(BASE_CHANNEL * 32, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      pix2pix.upsample(BASE_CHANNEL * 32, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
      pix2pix.upsample(BASE_CHANNEL * 32, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      pix2pix.upsample(BASE_CHANNEL * 32, 4, norm_type),  # (bs, 16, 16, 1024)
      pix2pix.upsample(BASE_CHANNEL * 16, 4, norm_type),  # (bs, 32, 32, 512)
      pix2pix.upsample(BASE_CHANNEL * 8, 4, norm_type),  # (bs, 64, 64, 256)
      pix2pix.upsample(BASE_CHANNEL * 4, 4, norm_type),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 4, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, 3])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def my_discriminator(norm_type='batchnorm', target=True):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
  Returns:
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  x = inp

  if target:
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)


  down1 = pix2pix.downsample(BASE_CHANNEL * 4, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
  down2 = pix2pix.downsample(BASE_CHANNEL * 8, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = pix2pix.downsample(BASE_CHANNEL * 16, 4, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(
      BASE_CHANNEL * 32, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  if target:
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  else:
    return tf.keras.Model(inputs=inp, outputs=last)

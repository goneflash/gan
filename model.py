from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

OUTPUT_CHANNELS = 3

def generator():
    return pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

def discriminator():
    return pix2pix.discriminator(norm_type='instancenorm', target=False)

# Usage:
#  python main.py --mode=train --dataset_path=/home/fan/dataset/celeb_img_align
#
#  python main.py --mode=predict \
#        --dataset_path=/home/fan/dataset/celeb_img_align --ckpt_path=./checkpoints/saved_ckpt
#
#  python main.py --mode=distill \
#        --dataset_path=/home/fan/dataset/celeb_img_align --ckpt_path=./checkpoints/saved_ckpt

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from model import generator, discriminator
from dataset_horse_zebra import get_horse_zebra_dataset
from dataset_male_female import get_male_female_dataset
from train import train_loop
from distill import distill_loop

import argparse
import os
import time
import sys
import getopt
import datetime
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

EPOCHS = 100
OUTPUT_CHANNELS = 3
BUFFER_SIZE = 1000
BATCH_SIZE = 4
MAX_NUM_SAMPLES = 50
NUM_SAMPLES_FOR_PREDICT = 50
MAX_CKPT_TO_SAVE = 10
NUM_EPOCHS_TO_SAVE = 5
# 512 must use batch size <= 2
IMG_WIDTH = 256
IMG_HEIGHT = 256

if __name__ == '__main__':
    try:
        optional_arguments = [
            'mode=', 'dataset_path=', 'ckpt_path=', 'dataset_name=',
            'distill_type=', 'batch_size=', 'max_num_samples='
        ]
        opts, args = getopt.getopt(sys.argv[1:], '', optional_arguments)
    except getopt.GetoptError:
        sys.exit('Usage: main.py --mode=<mode>')

    checkpoint_path = None
    dataset_path = None
    dataset_name = 'gender'
    distill_type = 'male2female'
    mode = 'train'
    batch_size = BATCH_SIZE
    max_num_samples = MAX_NUM_SAMPLES

    for opt, arg in opts:
        if opt in ("-m", "--mode"):
            mode = arg
            if mode not in [
                    'train', 'predict', 'distill', 'distill_prediction'
            ]:
                exit('Wrong mode: {}'.format(mode))
        if opt == '--ckpt_path':
            checkpoint_path = arg
        if opt == '--dataset_path':
            dataset_path = arg
        if opt == '--dataset_name':
            dataset_name = arg
        if opt == '--distill_type':
            distill_type = arg
        if opt == '--batch_size':
            batch_size = int(arg)
        if opt == '--max_num_samples':
            max_num_samples = int(arg)
    print('*********************************************')
    print('Mode is: {}'.format(mode))
    print('Dataset is: {}'.format(dataset_name))
    print('Dataset path: {}'.format(dataset_path))
    print('Batch size is: {}'.format(batch_size))
    print('Max num samples is: {}'.format(max_num_samples))
    print('*********************************************')

    img_width = IMG_WIDTH
    img_height = IMG_HEIGHT

    if dataset_name == 'gender':
        train_x, train_y, test_x, test_y = get_male_female_dataset(
            img_width,
            img_height,
            batch_size,
            BUFFER_SIZE,
            max_num_samples,
            dataset_path,
            skip_small_images=False,
            cache=False)
    elif dataset_name == 'horse':
        train_x, train_y, test_x, test_y = get_horse_zebra_dataset(
            batch_size, BUFFER_SIZE, max_num_samples)

    generator_g = generator(img_width=img_width, img_height=img_height)
    generator_f = generator(img_width=img_width, img_height=img_height)
    discriminator_y = discriminator(img_width=img_width, img_height=img_height)
    discriminator_x = discriminator(img_width=img_width, img_height=img_height)

    print('Generator:')
    print(generator_g.summary())
    print('Discriminator:')
    print(discriminator_x.summary())

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    ckpt = tf.train.Checkpoint(
        generator_g=generator_g,
        generator_f=generator_f,
        discriminator_x=discriminator_x,
        discriminator_y=discriminator_y,
        generator_g_optimizer=generator_g_optimizer,
        generator_f_optimizer=generator_f_optimizer,
        discriminator_x_optimizer=discriminator_x_optimizer,
        discriminator_y_optimizer=discriminator_y_optimizer)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode == 'train':
        if checkpoint_path == None:
            checkpoint_path = os.path.join('checkpoints', current_time)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=MAX_CKPT_TO_SAVE)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored: {}!!'.format(
                ckpt_manager.latest_checkpoint))

        train_loop(train_x,
                   train_y,
                   test_x,
                   test_y,
                   generator_g,
                   generator_f,
                   discriminator_x,
                   discriminator_y,
                   generator_g_optimizer,
                   generator_f_optimizer,
                   discriminator_x_optimizer,
                   discriminator_y_optimizer,
                   ckpt_manager,
                   batch_size=batch_size,
                   epochs=EPOCHS,
                   num_epochs_to_save=NUM_EPOCHS_TO_SAVE)

    elif mode == 'predict':
        if checkpoint_path == None:
            exit('Error: Please specify checkpoint path')
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=MAX_CKPT_TO_SAVE)
        if not ckpt_manager.latest_checkpoint:
            exit('Error: ckpt not exist for predict')

        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored: {}!!'.format(
            ckpt_manager.latest_checkpoint))

        index = 0
        for image_x in test_x.take(NUM_SAMPLES_FOR_PREDICT):
            fake_image_y = generator_g(image_x)
            for fake_image, original_image in zip(fake_image_y, image_x):
                image = np.concatenate(
                    (original_image.numpy(), fake_image.numpy()), axis=1)
                image = ((image + 1.0) * 127.5).astype(np.uint8)

                pil_img = Image.fromarray(image)
                file_name = os.path.join('./', 'output',
                                         'fake_image_y' + str(index) + '.png')
                pil_img.save(file_name)
                index += 1

        index = 0
        for image_y in test_y.take(NUM_SAMPLES_FOR_PREDICT):
            fake_image_x = generator_f(image_y)
            for fake_image, original_image in zip(fake_image_x, image_y):
                image = np.concatenate(
                    (original_image.numpy(), fake_image.numpy()), axis=1)
                image = ((image + 1.0) * 127.5).astype(np.uint8)

                pil_img = Image.fromarray(image)
                file_name = os.path.join('./', 'output',
                                         'fake_image_x' + str(index) + '.png')
                pil_img.save(file_name)
                index += 1

    elif mode == 'distill':
        if checkpoint_path == None:
            exit('Error: Please specify checkpoint path')
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=MAX_CKPT_TO_SAVE)
        if not ckpt_manager.latest_checkpoint:
            exit('Error: ckpt not exist for predict')

        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored: {}!!'.format(
            ckpt_manager.latest_checkpoint))

        tiny_generator = generator(3)
        tiny_generator.summary()

        distill_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        distill_ckpt = tf.train.Checkpoint(tiny_generator=tiny_generator,
                                           distill_optimizer=distill_optimizer)

        distill_ckpt_path = os.path.join('checkpoints_distill', current_time)
        distill_ckpt_manager = tf.train.CheckpointManager(
            distill_ckpt, distill_ckpt_path, max_to_keep=MAX_CKPT_TO_SAVE)

        if distill_type == 'male2female':
            train_dataset = train_x
            test_dataset = test_x
            original_generator = generator_g
        elif distill_type == 'female2male':
            train_dataset = train_y
            test_dataset = test_y
            original_generator = generator_f
        else:
            exit('Error: Unknown distill type')

        distill_loop(train_dataset,
                     test_dataset,
                     tiny_generator,
                     original_generator,
                     distill_optimizer,
                     distill_ckpt_manager,
                     batch_size=batch_size,
                     epochs=EPOCHS,
                     num_epochs_to_save=NUM_EPOCHS_TO_SAVE)

        # Convert to tflite as well.
        converter = tf.lite.TFLiteConverter.from_keras_model(tiny_generator)
        tflite_model = converter.convert()
        open('tflite/' + distill_type + '.tflite', "wb").write(tflite_model)

        # Also make some predictions
        for index, image_x in enumerate(
                test_dataset.take(NUM_SAMPLES_FOR_PREDICT)):
            original_model_output = original_generator(image_x)
            tiny_model_output = tiny_generator(image_x)

            image = np.concatenate(
                (image_x[0].numpy(), original_model_output[0].numpy(),
                 tiny_model_output[0].numpy()),
                axis=1)
            image = ((image + 1.0) * 127.5).astype(np.uint8)

            pil_img = Image.fromarray(image)

            file_name = os.path.join('./', 'output_tiny',
                                     'tiny_compare' + str(index) + '.png')
            pil_img.save(file_name)

    else:
        print('Error: Unknown mode {}'.format(mode))

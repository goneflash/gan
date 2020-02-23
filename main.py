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
import tensorflow_datasets as tfds
from model import generator, discriminator
from dataset_horse_zebra import get_horse_zebra_dataset
from dataset_male_female import get_male_female_dataset

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

EPOCHS = 30
OUTPUT_CHANNELS = 3
BUFFER_SIZE = 1000
BATCH_SIZE = 8
MAX_NUM_SAMPLES = 10000
NUM_SAMPLES_FOR_PREDICT=50
LAMBDA = 10


def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

@tf.function
def distill_train_step(input_image):
    with tf.GradientTape(persistent=True) as tape:
        original_output = generator_g(input_image, training=False)
        tiny_output = tiny_generator(input_image, training=True)
        simulate_loss = identity_loss(original_output, tiny_output)

        tiny_generator_train_loss(simulate_loss)

    tiny_generator_gradients = tape.gradient(simulate_loss, tiny_generator.trainable_variables)
    distill_optimizer.apply_gradients(zip(tiny_generator_gradients, tiny_generator.trainable_variables))

@tf.function
def distill_test_step(input_image):
    original_output = generator_g(input_image, training=False)
    tiny_output = tiny_generator(input_image, training=False)
    simulate_loss = identity_loss(original_output, tiny_output)

    tiny_generator_test_loss(simulate_loss)

@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
      # Generator G translates X -> Y
      # Generator F translates Y -> X.
      
      fake_y = generator_g(real_x, training=True)
      cycled_x = generator_f(fake_y, training=True)
  
      fake_x = generator_f(real_y, training=True)
      cycled_y = generator_g(fake_x, training=True)
  
      # same_x and same_y are used for identity loss.
      same_x = generator_f(real_x, training=True)
      same_y = generator_g(real_y, training=True)
  
      disc_real_x = discriminator_x(real_x, training=True)
      disc_real_y = discriminator_y(real_y, training=True)
  
      disc_fake_x = discriminator_x(fake_x, training=True)
      disc_fake_y = discriminator_y(fake_y, training=True)
  
      # calculate the loss
      gen_g_loss = generator_loss(disc_fake_y)
      gen_f_loss = generator_loss(disc_fake_x)
      
      total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
      
      # Total generator loss = adversarial loss + cycle loss
      total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
      total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
  
      generator_g_train_loss(total_gen_g_loss)
      generator_f_train_loss(total_gen_f_loss)
  
      disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
      disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
      discriminator_x_train_loss(disc_x_loss)
      discriminator_y_train_loss(disc_y_loss)
    
    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          generator_f.trainable_variables)
    
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                              discriminator_y.trainable_variables)
    
    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              generator_g.trainable_variables))
  
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                              generator_f.trainable_variables))
    
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
    
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

@tf.function
def test_step(real_x, real_y):
    fake_y = generator_g(real_x, training=False)
    cycled_x = generator_f(fake_y, training=False)

    fake_x = generator_f(real_y, training=False)
    cycled_y = generator_g(fake_x, training=False)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=False)
    same_y = generator_g(real_y, training=False)

    disc_real_x = discriminator_x(real_x, training=False)
    disc_real_y = discriminator_y(real_y, training=False)

    disc_fake_x = discriminator_x(fake_x, training=False)
    disc_fake_y = discriminator_y(fake_y, training=False)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    generator_g_test_loss(total_gen_g_loss)
    generator_f_test_loss(total_gen_f_loss)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    discriminator_x_test_loss(disc_x_loss)
    discriminator_y_test_loss(disc_y_loss)


if __name__ == '__main__':
    try:
        optional_arguments = ['mode=', 'dataset_path=', 'ckpt_path=', 'dataset_name=', 'distill_type=']
        opts, args = getopt.getopt(sys.argv[1:], '', optional_arguments)
    except getopt.GetoptError:
        sys.exit('Usage: main.py --mode=<mode>')

    checkpoint_path = None
    dataset_path = None
    dataset_name = 'gender'
    distill_type = 'male2female'
    mode = 'train'

    for opt, arg in opts:
        if opt in ("-m", "--mode"):
            mode = arg
            if mode not in ['train', 'predict', 'distill']:
                exit('Wrong mode: {}'.format(mode))
        if opt == '--ckpt_path':
            checkpoint_path = arg
        if opt == '--dataset_path':
            dataset_path = arg
        if opt == '--dataset_name':
            dataset_name = arg
        if opt == '--distill_type':
            distill_type = arg
    print('Mode is: {}'.format(mode))
    print('Dataset is: {}'.format(dataset_name))
    print('Dataset path: {}'.format(dataset_path))
    
    if dataset_name == 'gender':
        train_x, train_y, test_x, test_y = get_male_female_dataset(
                BATCH_SIZE, BUFFER_SIZE, MAX_NUM_SAMPLES, dataset_path)
    elif dataset_name == 'horse':
        train_x, train_y, test_x, test_y = get_horse_zebra_dataset(
                BATCH_SIZE, BUFFER_SIZE, MAX_NUM_SAMPLES)

    generator_g = generator()
    generator_f = generator()
    discriminator_y = discriminator()
    discriminator_x = discriminator()

    print('Generator:')
    print(generator_g.summary())
    print('Discriminator:')
    print(discriminator_x.summary())
    
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
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
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
          ckpt.restore(ckpt_manager.latest_checkpoint)
          print ('Latest checkpoint restored: {}!!'.format(ckpt_manager.latest_checkpoint))
    
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        generator_f_train_loss = tf.keras.metrics.Mean('generator_f_train_loss', dtype=tf.float32)
        generator_g_train_loss = tf.keras.metrics.Mean('generator_g_train_loss', dtype=tf.float32)
        discriminator_y_train_loss = tf.keras.metrics.Mean('discriminator_y_train_loss', dtype=tf.float32)
        discriminator_x_train_loss = tf.keras.metrics.Mean('discriminator_x_train_loss', dtype=tf.float32)
        generator_f_test_loss = tf.keras.metrics.Mean('generator_f_test_loss', dtype=tf.float32)
        generator_g_test_loss = tf.keras.metrics.Mean('generator_g_test_loss', dtype=tf.float32)
        discriminator_y_test_loss = tf.keras.metrics.Mean('discriminator_y_test_loss', dtype=tf.float32)
        discriminator_x_test_loss = tf.keras.metrics.Mean('discriminator_x_test_loss', dtype=tf.float32)
    
        generator_f_train_log_dir = 'logs/gradient_tape/' + current_time + '/generator_f_train'
        generator_g_train_log_dir = 'logs/gradient_tape/' + current_time + '/generator_g_train'
        generator_f_train_summary_writer = tf.summary.create_file_writer(generator_f_train_log_dir)
        generator_g_train_summary_writer = tf.summary.create_file_writer(generator_g_train_log_dir)
        discriminator_y_train_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_y_train'
        discriminator_x_train_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_x_train'
        discriminator_y_train_summary_writer = tf.summary.create_file_writer(discriminator_y_train_log_dir)
        discriminator_x_train_summary_writer = tf.summary.create_file_writer(discriminator_x_train_log_dir)
        generator_f_test_log_dir = 'logs/gradient_tape/' + current_time + '/generator_f_test'
        generator_g_test_log_dir = 'logs/gradient_tape/' + current_time + '/generator_g_test'
        generator_f_test_summary_writer = tf.summary.create_file_writer(generator_f_test_log_dir)
        generator_g_test_summary_writer = tf.summary.create_file_writer(generator_g_test_log_dir)
        discriminator_y_test_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_y_test'
        discriminator_x_test_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_x_test'
        discriminator_y_test_summary_writer = tf.summary.create_file_writer(discriminator_y_test_log_dir)
        discriminator_x_test_summary_writer = tf.summary.create_file_writer(discriminator_x_test_log_dir)
    
        for epoch in range(EPOCHS):
          start = time.time()
        
          for image_x, image_y in tf.data.Dataset.zip((train_x, train_y)):
            train_step(image_x, image_y)
          print ('Time taken for training epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
          start = time.time()

          for image_x, image_y in tf.data.Dataset.zip((test_x, test_y)):
            test_step(image_x, image_y)
          print ('Time taken for test epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
          
          fake_y = generator_g(image_x)
          fake_x = generator_f(image_y)
          with generator_f_train_summary_writer.as_default():
              tf.summary.scalar('generator_loss', generator_f_train_loss.result(), step=epoch)
          with generator_g_train_summary_writer.as_default():
              tf.summary.scalar('generator_loss', generator_g_train_loss.result(), step=epoch)
          with discriminator_y_train_summary_writer.as_default():
              tf.summary.scalar('discriminator_loss', discriminator_y_train_loss.result(), step=epoch)
          with discriminator_x_train_summary_writer.as_default():
              tf.summary.scalar('discriminator_loss', discriminator_x_train_loss.result(), step=epoch)

          with generator_f_test_summary_writer.as_default():
              tf.summary.scalar('generator_loss', generator_f_test_loss.result(), step=epoch)
              tf.summary.image("Input X", image_x, step=epoch)
              tf.summary.image("Faked Y", fake_y, step=epoch)
          with generator_g_test_summary_writer.as_default():
              tf.summary.scalar('generator_loss', generator_g_test_loss.result(), step=epoch)
              tf.summary.image("Input Y", image_y, step=epoch)
              tf.summary.image("Faked X", fake_x, step=epoch)
          with discriminator_y_test_summary_writer.as_default():
              tf.summary.scalar('discriminator_loss', discriminator_y_test_loss.result(), step=epoch)
          with discriminator_x_test_summary_writer.as_default():
              tf.summary.scalar('discriminator_loss', discriminator_x_test_loss.result(), step=epoch)
        
          if (epoch + 1) % 20 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))
        
          generator_f_train_loss.reset_states()
          generator_g_train_loss.reset_states()
          discriminator_y_train_loss.reset_states()
          discriminator_x_train_loss.reset_states()
          generator_f_test_loss.reset_states()
          generator_g_test_loss.reset_states()
          discriminator_y_test_loss.reset_states()
          discriminator_x_test_loss.reset_states()

          print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                              time.time()-start))


    elif mode == 'predict':
        if checkpoint_path == None:
            exit('Error: Please specify checkpoint path')
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if not ckpt_manager.latest_checkpoint:
            exit('Error: ckpt not exist for predict')

        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored: {}!!'.format(ckpt_manager.latest_checkpoint))

        for index, image_x in enumerate(test_x.take(NUM_SAMPLES_FOR_PREDICT)):
            fake_image_y = generator_g(image_x)
            image = np.concatenate((image_x[0].numpy(), fake_image_y[0].numpy()), axis=1)
            image = ((image + 1.0) * 127.5).astype(np.uint8)

            pil_img = Image.fromarray(image)
            file_name = os.path.join('./', 'output', 'fake_image_y' + str(index) + '.png')
            pil_img.save(file_name)

        for index, image_y in enumerate(test_y.take(NUM_SAMPLES_FOR_PREDICT)):
            fake_image_x = generator_f(image_y)
            image = np.concatenate((image_y[0].numpy(), fake_image_x[0].numpy()), axis=1)
            image = ((image + 1.0) * 127.5).astype(np.uint8)

            pil_img = Image.fromarray(image)
            file_name = os.path.join('./', 'output', 'fake_image_x' + str(index) + '.png')
            pil_img.save(file_name)
        
        generator_f.save('./saved_model/generator_f') 
        generator_g.save('./saved_model/generator_g') 
        discriminator_x.save('./saved_model/discriminator_x') 
        discriminator_y.save('./saved_model/discriminator_y') 
        generator_f.summary()

    elif mode == 'distill':
        if checkpoint_path == None:
            exit('Error: Please specify checkpoint path')
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if not ckpt_manager.latest_checkpoint:
            exit('Error: ckpt not exist for predict')

        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored: {}!!'.format(ckpt_manager.latest_checkpoint))

        tiny_generator = generator(3)
        tiny_generator.summary()

        distill_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        distill_ckpt = tf.train.Checkpoint(
                tiny_generator=tiny_generator,
                distill_optimizer=distill_optimizer)

        distill_ckpt_path = os.path.join('checkpoints_distill', current_time)
        distill_ckpt_manager = tf.train.CheckpointManager(distill_ckpt, distill_ckpt_path, max_to_keep=5)

        tiny_generator_train_loss = tf.keras.metrics.Mean('tiny_generator_train_loss', dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tiny_generator_train_log_dir = 'logs/gradient_tape/' + current_time + '/tiny_generator_train'
        tiny_generator_train_summary_writer = tf.summary.create_file_writer(tiny_generator_train_log_dir)
        tiny_generator_test_loss = tf.keras.metrics.Mean('tiny_generator_test_loss', dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tiny_generator_test_log_dir = 'logs/gradient_tape/' + current_time + '/tiny_generator_test'
        tiny_generator_test_summary_writer = tf.summary.create_file_writer(tiny_generator_test_log_dir)

        if distill_type == 'male2female':
            train_dataset = train_x
            test_dataset = test_x
        elif distill_type == 'female2male':
            train_dataset = train_y
            test_dataset = test_y
        else:
            exit('Error: Unknown distill type')

        for epoch in range(EPOCHS):
          start = time.time()
        
          for image_x in train_dataset:
            distill_train_step(image_x)
          print ('Time taken for training epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
          start = time.time()

          for image_x in test_dataset:
            distill_test_step(image_x)
          print ('Time taken for test epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

          if (epoch + 1) % 20 == 0:
            distill_ckpt_save_path = distill_ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 distill_ckpt_save_path))

          original_model_output = generator_g(image_x)
          tiny_model_output = tiny_generator(image_x)
          with tiny_generator_train_summary_writer.as_default():
              tf.summary.scalar('tiny_generator_loss', tiny_generator_train_loss.result(), step=epoch)
          with tiny_generator_test_summary_writer.as_default():
              tf.summary.scalar('tiny_generator_loss', tiny_generator_test_loss.result(), step=epoch)
              tf.summary.image("input X", image_x, step=epoch)
              tf.summary.image("original output X", original_model_output, step=epoch)
              tf.summary.image("tiny_model_output", tiny_model_output, step=epoch)

          tiny_generator_train_loss.reset_states()
          tiny_generator_test_loss.reset_states()
        tiny_generator.save('./saved_model/tiny_' + distill_type + '_generator') 

        # Convert to tflite as well.
        converter = tf.lite.TFLiteConverter.from_keras_model(tiny_generator)
        tflite_model = converter.convert()
        open('tflite/' + distill_type + '.tflite', "wb").write(tflite_model)

        # Also make some predictions
        for index, image_x in enumerate(test_x.take(NUM_SAMPLES_FOR_PREDICT)):
            original_model_output = generator_g(image_x)
            tiny_model_output = tiny_generator(image_x)

            image = np.concatenate((image_x[0].numpy(), original_model_output[0].numpy(), tiny_model_output[0].numpy()), axis=1)
            image = ((image + 1.0) * 127.5).astype(np.uint8)

            pil_img = Image.fromarray(image)

            file_name = os.path.join('./', 'output_tiny', 'tiny_compare' + str(index) + '.png')
            pil_img.save(file_name)

    else:
        print('Error: Unknown mode {}'.format(mode))

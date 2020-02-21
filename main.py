# Usage: python main.py --mode=predict --ckpt_path=./checkpoints/saved_ckpt
#        python main.py --mode=train
#        python main.py --mode=distill --ckpt_path=./checkpoints/saved_ckpt

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from model import generator, discriminator
from dataset import get_male_female_dataset, get_horse_zebra_dataset

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
BATCH_SIZE = 1
MAX_NUM_SAMPLES = 10
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
def distill_step(input_image):
    with tf.GradientTape(persistent=True) as tape:
        original_output = generator_g(input_image, training=False)
        tiny_output = tiny_generator(input_image, training=True)
        simulate_loss = identity_loss(original_output, tiny_output)

        tiny_generator_train_loss(simulate_loss)

    tiny_generator_gradients = tape.gradient(simulate_loss, tiny_generator.trainable_variables)
    distill_optimizer.apply_gradients(zip(tiny_generator_gradients, tiny_generator.trainable_variables))


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


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['mode=', 'ckpt_path='])
    except getopt.GetoptError:
        print('Usage: main.py --mode=<mode>')
        sys.exit(2)

    checkpoint_path = "./checkpoints/train"
    mode = 'train'

    for opt, arg in opts:
        if opt in ("-m", "--mode"):
            mode = arg
            if mode not in ['train', 'predict', 'distill']:
                print('Wrong mode: {}'.format(mode))
                exit()
        if opt == '--ckpt_path':
            checkpoint_path = arg
    print('{} mode'.format(mode))
    
    #train_horses, train_zebras = get_horse_zebra_dataset(BATCH_SIZE, BUFFER_SIZE, MAX_NUM_SAMPLES)
    train_horses, train_zebras = get_male_female_dataset(BATCH_SIZE, BUFFER_SIZE, MAX_NUM_SAMPLES)

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

    if mode == 'train':
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
    
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        generator_f_train_log_dir = 'logs/gradient_tape/' + current_time + '/generator_f_train'
        generator_g_train_log_dir = 'logs/gradient_tape/' + current_time + '/generator_g_train'
        generator_f_train_summary_writer = tf.summary.create_file_writer(generator_f_train_log_dir)
        generator_g_train_summary_writer = tf.summary.create_file_writer(generator_g_train_log_dir)
        discriminator_y_train_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_y_train'
        discriminator_x_train_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_x_train'
        discriminator_y_train_summary_writer = tf.summary.create_file_writer(discriminator_y_train_log_dir)
        discriminator_x_train_summary_writer = tf.summary.create_file_writer(discriminator_x_train_log_dir)
    
        for epoch in range(EPOCHS):
          start = time.time()
        
          for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
            train_step(image_x, image_y)
          
          fake_y = generator_g(image_x)
          fake_x = generator_f(image_y)
          with generator_f_train_summary_writer.as_default():
              tf.summary.scalar('generator_loss', generator_f_train_loss.result(), step=epoch)
              tf.summary.image("Input X", image_x, step=epoch)
              tf.summary.image("Faked Y", fake_y, step=epoch)
          with generator_g_train_summary_writer.as_default():
              tf.summary.scalar('generator_loss', generator_g_train_loss.result(), step=epoch)
              tf.summary.image("Input Y", image_y, step=epoch)
              tf.summary.image("Faked X", fake_x, step=epoch)
          with discriminator_y_train_summary_writer.as_default():
              tf.summary.scalar('discriminator_loss', discriminator_y_train_loss.result(), step=epoch)
          with discriminator_x_train_summary_writer.as_default():
              tf.summary.scalar('discriminator_loss', discriminator_x_train_loss.result(), step=epoch)
        
          if (epoch + 1) % 20 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))
        
          generator_f_train_loss.reset_states()
          generator_g_train_loss.reset_states()
          discriminator_y_train_loss.reset_states()
          discriminator_x_train_loss.reset_states()
          print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                              time.time()-start))
    elif mode == 'distill':
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if not ckpt_manager.latest_checkpoint:
            print('Error: ckpt not exist for predict')
            exit()

        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored: {}!!'.format(ckpt_manager.latest_checkpoint))

        tiny_generator = generator(4)
        tiny_generator.summary()
        distill_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        distill_ckpt = tf.train.Checkpoint(
                tiny_generator=tiny_generator,
                distill_optimizer=distill_optimizer)
        distill_ckpt_path = './checkpoints_distill'
        distill_ckpt_manager = tf.train.CheckpointManager(distill_ckpt, distill_ckpt_path, max_to_keep=5)

        tiny_generator_train_loss = tf.keras.metrics.Mean('tiny_generator_train_loss', dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tiny_generator_train_log_dir = 'logs/gradient_tape/' + current_time + '/tiny_generator_train'
        tiny_generator_train_summary_writer = tf.summary.create_file_writer(tiny_generator_train_log_dir)

        for epoch in range(EPOCHS):
          start = time.time()
        
          for image_x in train_horses:
            distill_step(image_x)
          if (epoch + 1) % 20 == 0:
            distill_ckpt_save_path = distill_ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 distill_ckpt_save_path))

          big_model_output = generator_g(image_x)
          tiny_model_output = tiny_generator(image_x)
          with tiny_generator_train_summary_writer.as_default():
              tf.summary.scalar('tiny_generator_loss', tiny_generator_train_loss.result(), step=epoch)
              tf.summary.image("input X", image_x, step=epoch)
              tf.summary.image("big output X", big_model_output, step=epoch)
              tf.summary.image("tiny_model_output", tiny_model_output, step=epoch)

          tiny_generator_train_loss.reset_states()
        tiny_generator.save('./saved_model/tiny_generator') 

        # Also make some predictions
        for index, horse in enumerate(train_horses.take(NUM_SAMPLES_FOR_PREDICT)):
            big_model_output = generator_g(horse)
            tiny_model_output = tiny_generator(horse)

            image = np.concatenate((horse[0].numpy(), big_model_output[0].numpy(), tiny_model_output[0].numpy()), axis=1)
            image = ((image + 1.0) * 127.5).astype(np.uint8)

            pil_img = Image.fromarray(image)

            file_name = os.path.join('./', 'output_tiny', 'tiny_compare' + str(index) + '.png')
            pil_img.save(file_name)


    elif mode == 'predict':
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if not ckpt_manager.latest_checkpoint:
            print('Error: ckpt not exist for predict')
            exit()

        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored: {}!!'.format(ckpt_manager.latest_checkpoint))

        for index, horse in enumerate(train_horses.take(NUM_SAMPLES_FOR_PREDICT)):
            fake_zebra = generator_g(horse)
            image = np.concatenate((horse[0].numpy(), fake_zebra[0].numpy()), axis=1)
            image = ((image + 1.0) * 127.5).astype(np.uint8)

            pil_img = Image.fromarray(image)

            file_name = os.path.join('./', 'output', 'fake_zebra' + str(index) + '.png')
            pil_img.save(file_name)

        for index, zebra in enumerate(train_zebras.take(NUM_SAMPLES_FOR_PREDICT)):
            fake_horse = generator_f(zebra)
            image = np.concatenate((zebra[0].numpy(), fake_horse[0].numpy()), axis=1)
            image = ((image + 1.0) * 127.5).astype(np.uint8)

            pil_img = Image.fromarray(image)

            file_name = os.path.join('./', 'output', 'fake_horse' + str(index) + '.png')
            pil_img.save(file_name)
        
        generator_f.save('./saved_model/generator_f') 
        generator_g.save('./saved_model/generator_g') 
        discriminator_x.save('./saved_model/discriminator_x') 
        discriminator_y.save('./saved_model/discriminator_y') 
        generator_f.summary()

        # Load model not working yet.
        #new_model = tf.keras.models.load_model('saved_model/generator_g')
        #batch_input_shape = (None, 256, 256, 3)
        #input_2.build(batch_input_shape)
        #new_model.summary()

    else:
        print('Error: Unknown mode {}'.format(mode))

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import datetime
import matplotlib.pyplot as plt
from IPython.display import clear_output

tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 40
OUTPUT_CHANNELS = 3
BUFFER_SIZE = 1000
BATCH_SIZE = 1
MAX_NUM_SAMPLES = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10

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

    discriminator_g_train_loss(total_gen_g_loss)
    discriminator_f_train_loss(total_gen_f_loss)
  
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
    print('Start')
    dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)
    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']
    train_horses = train_horses.map(
            preprocess_image_train, num_parallel_calls=AUTOTUNE).take(MAX_NUM_SAMPLES).cache().shuffle(BUFFER_SIZE).batch(1)

    train_zebras = train_zebras.map(
            preprocess_image_train, num_parallel_calls=AUTOTUNE).take(MAX_NUM_SAMPLES).cache().shuffle(BUFFER_SIZE).batch(1)

    test_horses = test_horses.map(
            preprocess_image_test, num_parallel_calls=AUTOTUNE).take(MAX_NUM_SAMPLES).cache().shuffle(BUFFER_SIZE).batch(1)

    test_zebras = test_zebras.map(
            preprocess_image_test, num_parallel_calls=AUTOTUNE).take(MAX_NUM_SAMPLES).cache().shuffle(BUFFER_SIZE).batch(1)


    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    
    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_f=generator_f,
                               discriminator_x=discriminator_x,
                               discriminator_y=discriminator_y,
                               generator_g_optimizer=generator_g_optimizer,
                               generator_f_optimizer=generator_f_optimizer,
                               discriminator_x_optimizer=discriminator_x_optimizer,
                               discriminator_y_optimizer=discriminator_y_optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_f_train_loss = tf.keras.metrics.Mean('generator_f_train_loss', dtype=tf.float32)
    generator_g_train_loss = tf.keras.metrics.Mean('generator_g_train_loss', dtype=tf.float32)
    discriminator_f_train_loss = tf.keras.metrics.Mean('discriminator_f_train_loss', dtype=tf.float32)
    discriminator_g_train_loss = tf.keras.metrics.Mean('discriminator_g_train_loss', dtype=tf.float32)

    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    generator_f_train_log_dir = 'logs/gradient_tape/' + current_time + '/generator_f_train'
    generator_g_train_log_dir = 'logs/gradient_tape/' + current_time + '/generator_g_train'
    generator_f_train_summary_writer = tf.summary.create_file_writer(generator_f_train_log_dir)
    generator_g_train_summary_writer = tf.summary.create_file_writer(generator_g_train_log_dir)
    discriminator_f_train_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_f_train'
    discriminator_g_train_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator_g_train'
    discriminator_f_train_summary_writer = tf.summary.create_file_writer(discriminator_f_train_log_dir)
    discriminator_g_train_summary_writer = tf.summary.create_file_writer(discriminator_g_train_log_dir)

    for epoch in range(EPOCHS):
      start = time.time()
    
      n = 0
      for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(image_x, image_y)
        if n % 10 == 0:
          print ('.', end='')
        n+=1
      
      fake_y = generator_g(image_x)
      fake_x = generator_f(image_y)
      with generator_f_train_summary_writer.as_default():
          tf.summary.scalar('generator_loss', generator_f_train_loss.result(), step=epoch)
      with generator_g_train_summary_writer.as_default():
          tf.summary.scalar('generator_loss', generator_g_train_loss.result(), step=epoch)
      with discriminator_f_train_summary_writer.as_default():
          tf.summary.scalar('discriminator_loss', discriminator_f_train_loss.result(), step=epoch)
      with discriminator_g_train_summary_writer.as_default():
          tf.summary.scalar('discriminator_loss', discriminator_g_train_loss.result(), step=epoch)
          #tf.summary.image("Input X", image_x, step=0)
          #tf.summary.image("Faked Y", fake_y, step=0)
          #tf.summary.image("Input Y", image_y, step=0)
          #tf.summary.image("Faked X", fake_x, step=0)
    
      if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))
    
      generator_f_train_loss.reset_states()
      generator_g_train_loss.reset_states()
      discriminator_f_train_loss.reset_states()
      discriminator_g_train_loss.reset_states()
      print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time()-start))

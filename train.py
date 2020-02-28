from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import time
import datetime

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_f_train_loss = tf.keras.metrics.Mean(
    'generator_f_train_loss', dtype=tf.float32)
generator_g_train_loss = tf.keras.metrics.Mean(
    'generator_g_train_loss', dtype=tf.float32)
discriminator_y_train_loss = tf.keras.metrics.Mean(
    'discriminator_y_train_loss', dtype=tf.float32)
discriminator_x_train_loss = tf.keras.metrics.Mean(
    'discriminator_x_train_loss', dtype=tf.float32)
generator_f_test_loss = tf.keras.metrics.Mean(
    'generator_f_test_loss', dtype=tf.float32)
generator_g_test_loss = tf.keras.metrics.Mean(
    'generator_g_test_loss', dtype=tf.float32)
discriminator_y_test_loss = tf.keras.metrics.Mean(
    'discriminator_y_test_loss', dtype=tf.float32)
discriminator_x_test_loss = tf.keras.metrics.Mean(
    'discriminator_x_test_loss', dtype=tf.float32)


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
def train_step(
        real_x,
        real_y,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        generator_g_optimizer,
        generator_f_optimizer,
        discriminator_x_optimizer,
        discriminator_y_optimizer):
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

        total_cycle_loss = calc_cycle_loss(
            real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + \
            total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + \
            total_cycle_loss + identity_loss(real_x, same_x)

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
def test_step(
        real_x,
        real_y,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        generator_g_optimizer,
        generator_f_optimizer,
        discriminator_x_optimizer,
        discriminator_y_optimizer):
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

    total_cycle_loss = calc_cycle_loss(
        real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + \
        total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + \
        total_cycle_loss + identity_loss(real_x, same_x)

    generator_g_test_loss(total_gen_g_loss)
    generator_f_test_loss(total_gen_f_loss)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    discriminator_x_test_loss(disc_x_loss)
    discriminator_y_test_loss(disc_y_loss)


def train_loop(
        train_x,
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
        batch_size=1,
        epochs=30,
        num_epochs_to_save=5):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root_dir = 'logs/gradient_tape/' + current_time
    generator_f_train_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/generator_f_train')
    generator_g_train_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/generator_g_train')
    discriminator_y_train_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/discriminator_y_train')
    discriminator_x_train_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/discriminator_x_train')
    generator_f_test_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/generator_f_test')
    generator_g_test_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/generator_g_test')
    discriminator_y_test_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/discriminator_y_test')
    discriminator_x_test_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/discriminator_x_test')

    for epoch in range(epochs):
        start = time.time()

        for image_x, image_y in tf.data.Dataset.zip((train_x, train_y)):
            train_step(
                image_x,
                image_y,
                generator_g,
                generator_f,
                discriminator_x,
                discriminator_y,
                generator_g_optimizer,
                generator_f_optimizer,
                discriminator_x_optimizer,
                discriminator_y_optimizer)

        fake_y = generator_g(image_x)
        fake_x = generator_f(image_y)
        with generator_f_train_summary_writer.as_default():
            tf.summary.scalar('generator_loss',
                              generator_f_train_loss.result(), step=epoch)
            tf.summary.image("train input X", image_x,
                             step=epoch, max_outputs=batch_size)
            tf.summary.image("train faked Y", fake_y,
                             step=epoch, max_outputs=batch_size)
        with generator_g_train_summary_writer.as_default():
            tf.summary.scalar('generator_loss',
                              generator_g_train_loss.result(), step=epoch)
            tf.summary.image("train input Y", image_y,
                             step=epoch, max_outputs=batch_size)
            tf.summary.image("train faked X", fake_x,
                             step=epoch, max_outputs=batch_size)
        with discriminator_y_train_summary_writer.as_default():
            tf.summary.scalar(
                'discriminator_loss', discriminator_y_train_loss.result(), step=epoch)
        with discriminator_x_train_summary_writer.as_default():
            tf.summary.scalar(
                'discriminator_loss', discriminator_x_train_loss.result(), step=epoch)
        print('Time taken for training epoch {} is {} sec\n'.format(
            epoch + 1, time.time()-start))
        start = time.time()

        for image_x, image_y in tf.data.Dataset.zip((test_x, test_y)):
            test_step(
                image_x,
                image_y,
                generator_g,
                generator_f,
                discriminator_x,
                discriminator_y,
                generator_g_optimizer,
                generator_f_optimizer,
                discriminator_x_optimizer,
                discriminator_y_optimizer)
        print('Time taken for test epoch {} is {} sec\n'.format(
            epoch + 1, time.time()-start))
        fake_y = generator_g(image_x)
        fake_x = generator_f(image_y)
        with generator_f_test_summary_writer.as_default():
            tf.summary.scalar('generator_loss',
                              generator_f_test_loss.result(), step=epoch)
            tf.summary.image("test input X", image_x,
                             step=epoch, max_outputs=batch_size)
            tf.summary.image("test faked Y", fake_y,
                             step=epoch, max_outputs=batch_size)
        with generator_g_test_summary_writer.as_default():
            tf.summary.scalar('generator_loss',
                              generator_g_test_loss.result(), step=epoch)
            tf.summary.image("test input Y", image_y,
                             step=epoch, max_outputs=batch_size)
            tf.summary.image("test faked X", fake_x,
                             step=epoch, max_outputs=batch_size)
        with discriminator_y_test_summary_writer.as_default():
            tf.summary.scalar('discriminator_loss',
                              discriminator_y_test_loss.result(), step=epoch)
        with discriminator_x_test_summary_writer.as_default():
            tf.summary.scalar('discriminator_loss',
                              discriminator_x_test_loss.result(), step=epoch)

        if (epoch + 1) % num_epochs_to_save == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

        generator_f_train_loss.reset_states()
        generator_g_train_loss.reset_states()
        discriminator_y_train_loss.reset_states()
        discriminator_x_train_loss.reset_states()
        generator_f_test_loss.reset_states()
        generator_g_test_loss.reset_states()
        discriminator_y_test_loss.reset_states()
        discriminator_x_test_loss.reset_states()

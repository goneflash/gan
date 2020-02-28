from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import time
import datetime

tiny_generator_train_loss = tf.keras.metrics.Mean(
    'tiny_generator_train_loss', dtype=tf.float32)
tiny_generator_test_loss = tf.keras.metrics.Mean(
    'tiny_generator_test_loss', dtype=tf.float32)


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return loss


@tf.function
def distill_train_step(
        input_image,
        tiny_generator,
        original_generator,
        distill_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        original_output = original_generator(input_image, training=False)
        tiny_output = tiny_generator(input_image, training=True)
        simulate_loss = identity_loss(original_output, tiny_output)

        tiny_generator_train_loss(simulate_loss)

    tiny_generator_gradients = tape.gradient(
        simulate_loss, tiny_generator.trainable_variables)
    distill_optimizer.apply_gradients(
        zip(tiny_generator_gradients, tiny_generator.trainable_variables))


@tf.function
def distill_test_step(
        input_image,
        tiny_generator,
        original_generator):
    original_output = original_generator(input_image, training=False)
    tiny_output = tiny_generator(input_image, training=False)
    simulate_loss = identity_loss(original_output, tiny_output)

    tiny_generator_test_loss(simulate_loss)


def distill_loop(
        train_dataset,
        test_dataset,
        tiny_generator,
        original_generator,
        distill_optimizer,
        distill_ckpt_manager,
        batch_size=1,
        epochs=30,
        num_epochs_to_save=5):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root_dir = 'logs/gradient_tape/' + current_time
    tiny_generator_train_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/tiny_generator_train')
    tiny_generator_test_summary_writer = tf.summary.create_file_writer(
        log_root_dir + '/tiny_generator_test')

    for epoch in range(epochs):
        start = time.time()

        # Train
        for image_x in train_dataset:
            distill_train_step(
                image_x,
                tiny_generator,
                original_generator,
                distill_optimizer)
        original_model_output = original_generator(image_x)
        tiny_model_output = tiny_generator(image_x)
        with tiny_generator_train_summary_writer.as_default():
            tf.summary.scalar('tiny_generator_loss',
                              tiny_generator_train_loss.result(), step=epoch)
            tf.summary.image("train input X", image_x, step=epoch)
            tf.summary.image("train original output X",
                             original_model_output, step=epoch)
            tf.summary.image("train tiny output",
                             tiny_model_output, step=epoch)
        print('Time taken for training epoch {} is {} sec\n'.format(
            epoch + 1, time.time()-start))
        start = time.time()

        # Evaluation
        for image_x in test_dataset:
            distill_test_step(
                image_x,
                tiny_generator,
                original_generator)
        print('Time taken for test epoch {} is {} sec\n'.format(
            epoch + 1, time.time()-start))

        if (epoch + 1) % num_epochs_to_save == 0:
            distill_ckpt_save_path = distill_ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                distill_ckpt_save_path))

        original_model_output = original_generator(image_x)
        tiny_model_output = tiny_generator(image_x)
        with tiny_generator_test_summary_writer.as_default():
            tf.summary.scalar('tiny_generator_loss',
                              tiny_generator_test_loss.result(), step=epoch)
            tf.summary.image("test input X", image_x, step=epoch)
            tf.summary.image("test original output X",
                             original_model_output, step=epoch)
            tf.summary.image("test tiny output",
                             tiny_model_output, step=epoch)

        tiny_generator_train_loss.reset_states()
        tiny_generator_test_loss.reset_states()

# import numpy as np
import tensorflow
import os
# from datetime import datetime
from loguru import logger
import time
# from IPython import display
import matplotlib.pyplot as plt


@tensorflow.function
def train_step(
    images,
    generator,
    discriminator,
    generator_loss,
    discriminator_loss,
    generator_optimizer,
    discriminator_optimizer,
    batch_size,
    noise_dim
):
    logger.debug("Batch size: " + str(batch_size))
    logger.debug("Noise dimention: " + str(noise_dim))
    noise = tensorflow.random.normal([batch_size, noise_dim])

    with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train(
    dataset, 
    epochs,
    generator,
    discriminator,
    generator_loss,
    discriminator_loss,
    generator_optimizer,
    discriminator_optimizer,
    checkpoint,
    checkpoint_prefix,
    seed,
    batch_size,
    noise_dim
):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            logger.debug(type(image_batch))
            logger.debug(len(image_batch))
            logger.debug(image_batch)
            train_step(
                image_batch,
                generator,
                discriminator,
                generator_loss,
                discriminator_loss,
                generator_optimizer,
                discriminator_optimizer,
                batch_size,
                noise_dim
            )

        # Produce images for the GIF as you go
        # display.clear_output(wait=True)
        # generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        logger.info('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                         epochs,
    #                         seed)
    predictions = generator(seed, training=False)
    img = tensorflow.keras.utils.array_to_img(predictions[0] * 255., scale=False)
    img.save(os.path.join('', 'WEDNESDAY_FROG.png'))



# def train_networks(latent_dim, generator, discriminator, gan, x_train, iterations, batch_size, save_dir, start):
#     for step in range(iterations):
#         random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

#         generated_images = generator.predict(random_latent_vectors)

#         stop = start + batch_size
#         real_images = x_train[start: stop]
#         combined_images = np.concatenate([generated_images, real_images])

#         labels = np.concatenate(
#             [np.ones((batch_size, 1)),
#             np.zeros((batch_size, 1))])
        
#         labels += 0.55 * np.random.random(labels.shape)

#         # Note: train_on_batch is a simple gradient loss
#         discriminator_loss = discriminator.train_on_batch(combined_images, labels)

#         random_latent_vectors = np.random.normal(size=(batch_size, latent_dim)) 

#         misleading_targets = np.zeros((batch_size, 1))

#         generator_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

#         start += batch_size
#         if start > len(x_train) - batch_size:
#             start = 0

#         if step % 100 == 0:
#             gan.save_weights('gan_model\gan.h5')

#             logger.info('Discriminator lossin step %s: %s' % (step, discriminator_loss))
#             logger.info('Generator loss in step %s: %s' % (step, generator_loss))
            
#             img = tensorflow.keras.utils.array_to_img(generated_images[0] * 255., scale=False)
#             img.save(os.path.join(save_dir, 'generated_frog_' + str(step) + '.png'))

#             img = tensorflow.keras.utils.array_to_img(real_images[0] * 255., scale=False)
#             img.save(os.path.join(save_dir, 'real_frog_' + str(step) + '.png'))
        
#         if step % 10 == 0:
#             now = datetime.now()
#             logger.info(
#                 now.strftime("%H:%M:%S") + " : step " + 
#                 str(step) + " out of " + str(iterations) + " steps (" + str((step*100)/iterations) + "%)"
#             )

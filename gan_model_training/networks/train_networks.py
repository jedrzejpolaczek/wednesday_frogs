import numpy as np
import tensorflow
import os
from datetime import datetime
from loguru import logger


def train_networks(latent_dim, generator, discriminator, gan, x_train, y_train, iterations, batch_size, save_dir, start):
    for step in range(iterations):
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        generated_images = generator.predict(random_latent_vectors)

        stop = start + batch_size
        real_images = x_train[start: stop]
        combined_images = np.concatenate([generated_images, real_images])

        labels = np.concatenate(
            [np.ones((batch_size, 1)),
            np.zeros((batch_size, 1))])
        
        labels += 0.55 * np.random.random(labels.shape)

        # Note: train_on_batch is a simple gradient loss
        discriminator_loss = discriminator.train_on_batch(combined_images, labels)

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim)) 

        misleading_targets = np.zeros((batch_size, 1))

        generator_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0

        if step % 100 == 0:
            gan.save_weights('gan_model\gan.h5')

            logger.info('Discriminator lossin step %s: %s' % (step, discriminator_loss))
            logger.info('Generator loss in step %s: %s' % (step, generator_loss))
            
            img = tensorflow.keras.utils.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_frog_' + str(step) + '.png'))

            img = tensorflow.keras.utils.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_frog_' + str(step) + '.png'))
        
        now = datetime.now()
        logger.info(now.strftime("%H:%M:%S") + " : step: " + str(step))

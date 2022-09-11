import keras
from keras import layers
import numpy as np
import os
import tensorflow
from datetime import datetime

print(tensorflow.__version__)
# Zmienne opisujące obrazy
latent_dim = 32
height = 32
width = 32
channels = 3

# -----------------
# --- GENERATOR ---
# -----------------

# INPUT LAYER
generator_input = keras.Input(shape=(latent_dim,))

# HIDDEN LAYERS
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((32, 32, 32))(x)  # Tworzymy 

x = layers.Conv2D(256, 5, padding='same')(x)
X = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 4, strides=1, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# OUTPUT LAYER
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

# DEKLARACJA MODELU GENERATORA
generator = keras.models.Model(generator_input, x)

# OPTYMALIZACJA MODELU GENERATORA
# N/A
# KOMPILACJA MODELU GENERATORA
# N/A

# ---------------------
# --- DYSKRYMINATOR ---
# ---------------------

# INPUT LAYER
discriminator_input = layers.Input(shape=(height, width, channels))

# HIDDEN LAYERS
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

# OUTPUT LAYER
x = layers.Dense(1, activation='sigmoid')(x)

# DEKLARACJA MODELU DYSKRYMINATORA
discriminator = keras.models.Model(discriminator_input, x)

# OPTYMALIZACJA MODELU DYSKRYMINATORA
discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8
)

# KOMPILACJA MODELU DYSKRYMINATORA
discriminator.compile(
    optimizer=discriminator_optimizer,
    loss='binary_crossentropy'
)

# -----------
# --- GAN ---
# -----------

discriminator.trainable = False

# INPUT LAYER
gan_input = keras.Input(shape=(latent_dim,))

# HIDDEN LAYERS
pass

# OUTPUT LAYER
gan_output = discriminator(generator(gan_input))

# DEKLARACJA MODELU GAN
gan = keras.models.Model(gan_input, gan_output)

# OPTYMALIZACJA MODELU GAN
gan_optimizer = keras.optimizers.RMSprop(
    lr=0.0004,
    clipvalue=1.0,
    decay=1e-8
)

# KOMPILACJA MODELU GAN
gan.compile(
    optimizer=gan_optimizer,
    loss='binary_crossentropy'
)

# ------------------
# --- TRENOWANIE ---
# ------------------

(x_train, y_train), (_, _) = tensorflow.keras.datasets.cifar10.load_data()  # Ładowanie zbioru CIFAR10

x_train = x_train[y_train.flatten() == 6]  # Wybór obrazów żab (klasa numer 6)

x_train = x_train.reshape(
    (x_train.shape[0],) +
    (height, width, channels)
).astype('float32') / 255

iterations = 10000
batch_size = 20
save_dir = "gan_images"
start = 0

for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))  # Próbkowanie losowych punktów z niejawnej przestrzeni

    generated_images = generator.predict(random_latent_vectors)  # Dekodowanie punktów w celu wygenerowania sztucznych obrazów

    # Łaczenie obrazów sztucznych z prawdziwymi
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    # Tworzenie etykiet umożliwiających odróżnienie obrazów prawdziwych od sztucznych
    labels = np.concatenate(
        [np.ones((batch_size, 1)),
        np.zeros((batch_size, 1))])
    
    labels += 0.55 * np.random.random(labels.shape)  # WAŻNE: wprowadzanie losowego szumu etykiet

    # Trenowanie dyskryminatora
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # Losowe próbkowanie punktów stwierdzających oryginalność przestrzeni
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim)) 

    # Tworzenie fałszywych etykiet stwierdzających oryginalność obrazów
    misleading_targets = np.zeros((batch_size, 1))

    # Trenowanie generatora przy użyciu modelu gan i zamrożeniu dyskryminatora
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    # Okazjonalny zapis danych i generowanie wykresów (co 100 kroków algorytmu)
    if step % 100 == 0:
        gan.save_weights('gan.h5')  # Zapis wag modelu

        print(step)
        print('Strata dysryminatora w kroku %s: %s' % (step, d_loss))
        print('strata przeciwna: %s: %s' % (step, a_loss))
        
        # Zapis jednego wygenerowanego obrazu
        img = tensorflow.keras.utils.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog_' + str(step) + '.png'))

        # Zapis jednego prawdziwego obrazu w celach porównawczych
        img = tensorflow.keras.utils.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog_' + str(step) + '.png'))
    
    now = datetime.now()
    print(now.strftime("%H:%M:%S") + " : step: " + str(step))
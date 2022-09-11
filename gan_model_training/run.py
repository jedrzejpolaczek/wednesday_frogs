import tensorflow

from networks.generator import create_generator
from networks.discriminator import create_discriminator
from networks.gan import create_gan
from networks.train_networks import train_networks
from datasets.cifar10_frogs import get_frogs


print(tensorflow.__version__)
# First fings first, lets check if we will be using GPU
if tensorflow.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tensorflow.test.gpu_device_name()))
else:
   print("Please install GPU version of tensorflow!")

# -----------------------
# ------ VARIABLES ------
# -----------------------

latent_dim = 32
height = 32
width = 32
channels = 3

iterations = 10000
batch_size = 20
save_dir = "gan_model_training\gan_images"
start = 0

# -----------------------
# --- CREATE NETWORKS ---
# -----------------------

generator = create_generator(latent_dim, channels)
discriminator = create_discriminator(height, width, channels)
gan = create_gan(discriminator, generator, latent_dim)

# -----------------------
# ---- DOWNLOAD DATA ----
# -----------------------

x_train, y_train = get_frogs(height, width, channels)

# -----------------------
# ---- TRAIN NETWORK ----
# -----------------------

train_networks(latent_dim, generator, discriminator, gan, x_train, y_train, iterations, batch_size, save_dir, start)

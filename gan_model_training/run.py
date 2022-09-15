import tensorflow

from networks.generator import create_generator
from networks.discriminator import create_discriminator
from networks.gan import create_gan
from networks.train_networks import train_networks
from datasets.cifar10_frogs import get_frogs
from utils.configuration import get_json_data


print(tensorflow.__version__)
# First fings first, lets check if we will be using GPU
if tensorflow.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tensorflow.test.gpu_device_name()))
else:
   print("Please install GPU version of tensorflow!")

# -----------------------
# ------ VARIABLES ------
# -----------------------

training_data = get_json_data('training_config.json')

latent_dim = training_data["latent_dim"]
height = training_data["height"]
width = training_data["width"]
channels = training_data["channels"]

iterations = training_data["iterations"]
batch_size = training_data["batch_size"]
save_dir = training_data["save_dir"]
start = training_data["start"]

# -----------------------
# --- CREATE NETWORKS ---
# -----------------------

generator = create_generator(latent_dim, height, width, channels)
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

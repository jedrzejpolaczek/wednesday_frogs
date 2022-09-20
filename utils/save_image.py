import os
import tensorflow

def save_image(save_dir, name, images):
    """ TODO: add docstring"""
    img = tensorflow.keras.utils.array_to_img(images[0] * 255., scale=False)
    img.save(os.path.join(save_dir, f'{name}.png'))

import os
import tensorflow

def generate_and_save_images(model, test_input, save_dir, name):
    """ TODO: add docstring"""
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    img = tensorflow.keras.utils.array_to_img(predictions[0] * 255., scale=False)
    img.save(os.path.join(save_dir, f'{name}.png'))

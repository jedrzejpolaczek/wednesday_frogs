def generate_images(model, test_input):
    """ TODO: add docstring"""
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictited_images = model(test_input, training=False)

    return predictited_images

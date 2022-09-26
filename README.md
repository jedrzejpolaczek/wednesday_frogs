# It's Wednesday my dudes!
Fun project for getting new, unique frog every Wednesday my dudes!

Warning: Basic idea for these model is inspired or based on implementation of:
1. GAN presented in `Deep Learning with Python` by `Francois Chollet`.
2. Tensorflow GAN example implementation.
* https://www.tensorflow.org/hub/tutorials/bigbigan_with_tf_hub
3. RealPython (https://realpython.com/how-to-make-a-discord-bot-python/)
4. Machine Learning mastery (https://machinelearningmastery.com):
* https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/?fbclid=IwAR0gqQEwPCUujK98NmE6-wBBPIVp5vFaouG3q0KiOVJKSojWtoHoFKeaN54
* https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/
5. Hacks and tips about GAN (https://github.com/soumith/ganhacks)
6. Stack Overflow (https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc)
7. ... and a lot more.

# Main concept
1. Frog database is CIFAR10.
2. Create and train generative adversarial networks (GAN) for generating frog images.
3. Script for generating and sending emails with frog image each Wednesday.
4. Run discord bot that will run script for generating and post frog image each Wednesday.
5. Everything is nicely documented (WIP)

# How it works?
Simplified concept will looks like that:
1. Generator tries to "predict" how an image should look in the same manner other neural networks try to predict the next word or if an image is a cat or dog. 
2. On the other hand, the discriminator is trying to simply predict if a given image is the original image from the data set or generated one.
3. Having a generator and discriminator we can put them together in one "loop training" network and in that way we get GAN!

# Types of GANs (WIP)
In catalogue `model` you can find diffrent implementation of GAN. For now you can find:
* DCGAN (catalogue `dcgan`) - Deep Convolutional Generative Adversarial Network, just GAN but network layers are mostly convolutional layers.
* (WIP) PGGAN (catalogue `pggan`) - Progressive Growing Generative Adversarial Network is GAN that can take smaller image as input (during training) and predict much more bigger image.


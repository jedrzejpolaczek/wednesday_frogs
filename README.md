# It's Wednesday my dudes!
Fun project for getting new, unique frog every Wednesday my dudes!

Warning: Basic idea for these model is inpired or based on implementation of:
1. GAN presented in `Deep Learning with Python` by `Francois Chollet`.
2. Tensorflow GAN example implementation.
3. RealPython (https://realpython.com/how-to-make-a-discord-bot-python/)
4. ... and a lot more.

# Main concept (WIP)
1. Frog database is CIFAR10.
2. Create and train generative adversarial networks (GAN) for generating frog images.
3. Script for generating and sending emails with frog image each Wednesday.
4. Run discord bot that will run script for generating and post frog image each Wednesday. (WIP)

# How it works?
Simplified concept will looks like that:
1. Generator tries to "predict" how an image should look in the same manner other neural networks try to predict the next word or if an image is a cat or dog. 
2. On the other hand, the discriminator is trying to simply predict if a given image is the original image from the data set or generated one.
3. Having a generator and discriminator we can put them together in one "loop training" network and in that way we get GAN!

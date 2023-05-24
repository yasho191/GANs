# GAN Implementations using PyTorch

This repository contains a collection of Generative Adversarial Network (GAN) implementations using PyTorch. GANs are a popular class of deep learning models that learn to generate realistic synthetic data by pitting a generator network against a discriminator network in a two-player adversarial game.

## Table of Contents

- [Introduction](#introduction)
- [Implemented GANs](#implemented-gans)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

## Introduction

In this repository, we provide PyTorch implementations of various GAN architectures. Each implementation includes the generator, discriminator, and training loop, code necessary to train and generate samples from the respective GAN model.

By studying and experimenting with these implementations, you can gain a better understanding of GANs and how they can be implemented using PyTorch. Additionally, you can use these implementations as a starting point for your own GAN projects and modify them to suit your specific requirements.

## Implemented GANs

The following GAN architectures have been implemented in this repository:

- Implemented Networks
    1. Deep Convolutional GAN (DCGAN)
    2. Conditional GAN (CGAN)

- Upcoming Nteworks
    1. Wasserstein GAN (WGAN)
    2. WGAN with Gradient Penalty (WGAN-GP)
    3. CycleGAN
    4. InfoGAN
    5. Style GAN

Each implementation is contained within its own directory.

## Getting Started

Refer to the main README.md file for setting up the project.

## Dependencies

The following dependencies are required to run the GAN implementations:

- Python 3.9>=
- PyTorch
- Torchvision
- NumPy
- Matplotlib

You can install the required Python packages by running the following command:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions to this repository are welcome! If you have any improvements, bug fixes, or additional GAN implementations you would like to contribute, please follow the standard GitHub workflow:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make the necessary modifications.
4. Commit your changes.
5. Push your changes to your forked repository.
6. Submit a pull request to the original repository.

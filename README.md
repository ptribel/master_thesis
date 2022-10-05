# Master thesis
This repository contains the data/code/report of my master thesis.

## Dataset
- A dataset is stored on the ULB OneDrive at this adress: https://universitelibrebruxelles-my.sharepoint.com/:f:/g/personal/pascal_tribel_ulb_be/Elxjlsx-Ee5GnouOrZNSZYMB2YSUuBff7Pj7-7EbG4vqjQ?e=LUfUgn
- This adress is only reacheable when connected to an ulb account
- It consists in three folders:
  - One with the clean samples
  - One with the samples with vinyl noise
  - One with the samples with white noise
- The wav files can be converted to a tensorflow dataset using the utils files in the `Generation` folder

- For now, the used data are the three wav file in this repository

## Models
For now, two models are available:
- An autoencoder
- A convolutional autoencoder
- A Generative Adversarial Network

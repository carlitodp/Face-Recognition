# Introduction

In this project, we implement a CNN-based Siamese network for face verification using TensorFlow. We train the model on the CelebA dataset, which contains over 200,000 images of more than 10,000 unique identities. The following sections detail the complete workflow, including the generation of positive and negative pairs, the model architecture, the evaluation metrics, the loss functions, and the experimental results.

# Dataset Download and Pair forming

The CelebA dataset can be downloaded following this link: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html or directly loaded from Tensorflow (https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data). In this case, the dataset was downloaded and with the Identities annotations file, each photo was matched to a unique ID.




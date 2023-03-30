# Intro of scFSNN
Feature selection based on deep neural network for scRNAseq data

# Getting started
In order to run scFSNN, an operative version of Python and TensorFlow is needed. The code has been tested on Python 3.8.12 and PyTorch 1.10.2. Other Python dependencies include numpy and scikit-learn.

# scFSNN

## Description
*scFSNN* is used to select features based on deep neural network for scRNAseq data.

## Usuage

scFSNN(train_X, train_Y,  val_X, val_Y, test_X, test_Y, 
         num_classes, lr, epochs, batch_size, q0, device,
         dropout=0.5, eta=1, elimination_rate=1, cut_off=0.1)

## Arguments

1. train_X training inputs
2. train_Y training output

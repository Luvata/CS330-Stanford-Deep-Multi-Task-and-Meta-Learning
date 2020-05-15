# HW1: Meta learning with MANN

[Original](http://cs330.stanford.edu/material/hw1_updated.zip) with omniglot dataset

## TODO
- [x] Problem 1: Data Processing for Few-Shot Classification
- [x] Problem 2: Memory Augmented Neural Networks [2, 3]
    - [x] Implement in Pytorch
    - [x] Fixed pytorch training error
    - [x] Added support cuda and tensorboard for pytorch
    - [x] Fix TF implementation error
- [x] Problem 3: Analysis
- [ ] Problem 4: Experimentation

~~- By using CNN as feature extractor and adding dropout, I got accuracy ~ 0.7 on 1 shot 5 ways~~ Turn out this 
overwhelming result comes from my bug that I accidentally zero-ing all meta-test labels of `input_labels` in `predict`, then calculate loss 
on it :scream: 


## Problem 3: Training result 

#### Training 
Follow this [colab](https://colab.research.google.com/drive/1bVR_v0bajtTdNazytj4ai2W6sfD7p6Kr?usp=sharing) for training
and visualize in tensorboard
![Training](output/tensorboard.png)


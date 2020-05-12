# HW1: Meta learning with MANN

[Original](http://cs330.stanford.edu/material/hw1_updated.zip) with omniglot dataset

- [Update 05/13]: Fixed abnormal training result with shuffle separate strategy :beetle: 
- [Update 04/11]: Revert `load_data` to previous shuffle strategy since new shuffle somehow won't work
- [Update 04/02]: Change `load_data` to shuffle separately the train and test set
- [Update 03/20]: Added `hw1_pytorch_cnn` with CNN encoder, I got accuracy ~ 0.7 on 1 shot 5 ways, but turn out the 
training is pretty unstable, my suggest is to use small batch size with more training steps

## TODO
Since I'm not familiar with TF, I try to implement Problem 2 in pytorch. I also loaded all images into RAM for faster IO

I should move on to the next assignment, so these refactors and cuda support will arrive later 

- [x] Problem 1: Data Processing for Few-Shot Classification
- [x] Problem 2: Memory Augmented Neural Networks [2, 3]
    - [x] Implement in Pytorch
    - [x] Fixed pytorch training error
    - [ ] Fix TF implementation error
    - [ ] ~~Refactor DataLoader~~
    - [ ] ~~Add support cuda~~
    - [ ] ~~Refactor episode phase and predict phase~~
- [ ] Problem 3: Analysis
- [ ] Problem 4: Experimentation

~~- By using CNN as feature extractor and adding dropout, I got accuracy ~ 0.7 on 1 shot 5 ways~~ Turn out this 
overwhelming result comes from my bug that I accidentally zero-ing all meta-test labels of `input_labels` in `predict`, then calculate loss 
on it :scream: 

Below is the actual result of my correct implementation 
`python hw1_pytorch.py --num_classes=2 --num_samples=1 --meta_batch_size=64`
![image](output/k1n2.png)

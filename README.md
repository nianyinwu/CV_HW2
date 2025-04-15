# NYCU Computer Vision 2025 Spring HW2
StudentID: 313551078 \
Name: 吳年茵

## Introduction
In this lab, we have to train a __Faster R-CNN__ as our model to solve a __digit recognition task__ with 11 classes (including background).

This lab has two tasks: 
1. Task 1 :  Detect each digit in the image.
2. Task 2 : Recognize the entire digit in the image. 

We use a dataset of 30,062 RGB images
for training, 3,034 for validation, and 13,068 for testing.
To improve the model’s performance for two tasks, I adopt __Faster R-CNN v2__ in pytorch, which modified the backbone and detection module to have better metrics than Faster R-CNN. I also use pre-trained weights from training on COCO_V1 to converge better and learn current training data based on previously learned visual features. 

Additionally, I experiment with some tricks, such as making some __data augmentations__ or introducing the __Convolutional Block Attention Module (CBAM)__, hope to improve the model’s performance. 


## How to install
1. Clone this repository and navigate to folder
```shell
git clone https://github.com/nianyinwu/CV_HW2.git
cd CV_HW2
```
2. Install environment
```shell
conda env create --file hw2.yml --force
conda activate hw2
```

## Training
```shell
cd codes
python3 train.py -e <epochs> -b <batch size> -lr <learning rate> -d <data path> -s <save path> 
```
## Testing ( Inference )
The two predicted results (pred.csv and pred.json) will be saved in the argument of save path .
```shell
cd codes
python3 inference.py -d <data path> -w <the path of model checkpoints> -s <save path>
```

## Performance snapshot
![image](https://github.com/nianyinwu/CV_HW1/blob/main/result/snapshot.png)
# D-ADD: Defense Against Model Stealing Based on Account-Aware Distribution Discrepancy

This repository contains the implementation of **D-ADD**, a detector for defence against model stealing attack.

---

## Getting Started 

Here is an example for for WRN16_4 and CIFAR10


### 1. Train a Clean Model
To train a model:
```bash
python train_model.py --model wrn16_4 --dataset cifar10 --epoch 50 --batch_size 256 --lr 0.1 --save_path pretrain/wrn16_4_cifar100.pth --data_path ../datasets --device cuda
```


### 2. Calculate Mean and Covariance
Calculate the mean and covariance of the sample features of the training set:
```bash
python get_mean_cov.py --model wrn16_4 --dataset cifar10 --num_classes 10 --data_path ../datasets --model_path pretrain --device cuda
```


### 3. Calculate Distance
Calculate the distance of dataset and target dataset:
```bash
python get_distance.py --model wrn16_4 --dataset cifar10 --target_dataset cifar10 --num_classes 10 --window_size 16 --data_path ../datasets --device cuda
python get_distance.py --model wrn16_4 --dataset cifar100 --target_dataset cifar10 --num_classes 10 --window_size 16 --data_path ../datasets --device cuda
```


### 4. Run Knockoff
Run knockoff
```bash
python knockoff.py --victim_model wrn16_4 --surrogate_model wrn16_4 --target_dataset cifar10 --surrogate_dataset cifar100 --train_set True --num_classes 10 --window_size 16 --threshold 11.6 --lr 0.1 --e 30 --data_path ../datasets --device cuda
```

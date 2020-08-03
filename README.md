# LLSQ-Unofficial-
The PyTorch implementation of Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware (LLSQ) in ICLR2020 (unofficial)

I am working on reproducing this paper. This project is based on the [code](https://anonymous.4open.science/r/c05a5b6a-1d0c-4201-926f-e7b52034f7a5/).

# Experiment

----- VGG-small Cifar10 Accuracy -----
|    |This Project| Paper    |
|--- |------------|----------|
|fp32|88.47 %     |93.34 %    |
|w4a4|90.25 %     |94.34 %    |

- fp32 training 400 epochs
- w4a4 training 100 epochs (Based on fp32)

# Reference
```
@inproceedings{
Zhao2020Linear,
title={Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware},
author={Xiandong Zhao and Ying Wang and Xuyi Cai and Cheng Liu and Lei Zhang},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1lBj2VFPS}
}
```

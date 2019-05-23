# DeepCompression

This is own implementation and understanding of the paper [DeepCompression](https://arxiv.org/abs/1510.00149) developed in June/July 2017 (except huffmann coding =D ), and very simple reports were produced on November.

## My tasks to solve
Learn Pytorch for low-level (gradient modification) and high-level implementation (large networks).
Learn a very efficient network optimization (2015). Currently, there is a new optimization method that I would like to learn =) [MorphNet](https://ai.googleblog.com/2019/04/morphnet-towards-faster-and-smaller.html)

## Update results
Trained models were based on VGG19 and custom versions, however results presented here are VGG19 (with batchnorm) trained on CIFAR10.

### Results

Results reported on VGG19 in this repository using k clusters keep almost the same accuracy same as the original paper argued. All optimization results were trained in an overall of 25 epochs. Pruning iteration number was to set 25 as well (just for convenience). 

| Netwrok   | Original  | Pruned 25 | Shared k=4 | Shared k=9 | Shared k=13 | Shared k=35 |
| --------- | --------- | --------- | ---------- | ---------- | ----------- | ----------- |
| VGG19_BN  |   92.22   |   92.18   |     90.93  |     91.86  |     92.23   |  (soon =D)  |

Some visual results of the pruned weights are shown following:

| Layer 00   | Layer 02  | Layer 06 | Layer 18 |
| ![Layer 00](https://github.com/emedinac/LearningAnimations/blob/master/Figures/Weight_00.png)   | ![Layer 02](https://github.com/emedinac/LearningAnimations/blob/master/Figures/Weight_02.png)  | ![Layer 06](https://github.com/emedinac/LearningAnimations/blob/master/Figures/Weight_06.png) |
| Layer 18  | Layer 31 | Layer 32 |
![Layer 18](https://github.com/emedinac/LearningAnimations/blob/master/Figures/Weight_18.png)  | ![Layer 31](https://github.com/emedinac/LearningAnimations/blob/master/Figures/Weight_31.png) | ![Layer 32](https://github.com/emedinac/LearningAnimations/blob/master/Figures/Weight_32.png) |





<!-- /wp:heading -->

## Release
Code now work for pytorch 1.0.1.
Trained models differ from the original ones reported in the presentation.




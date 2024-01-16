## ResNet

> [Deep Residual Learning for Image Recognitionn](https://arxiv.org/abs/1512.03385)

## Abstract

<!-- [ABSTRACT] -->

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

<!-- [IMAGE] -->

## Results and models

### CIFAR-10


| Model | Validation Accuracy | config | download    |
| :--------- | --------------------- | :------- | ------------- |
| ResNet-34  | 94.36%              | config | [model](https://drive.google.com/file/d/1-KAR6tC6wZNvkKL5C105t5VdxDmRvXt8/view?usp=sharing) / [log](https://drive.google.com/file/d/1-edPcaBcgVK0Bs4J1pB1HW91f0uFN_PT/view?usp=sharing) |
| ResNet-50  | 93.36%              | config | [model](https://drive.google.com/file/d/1-ZhMNGqGFdS2TRTG1hn3qgSzkBksYV5t/view?usp=sharing) / [log](https://drive.google.com/file/d/1txbkjLfDu4d2bVTwidn_vlA6rSny7-Qc/view?usp=sharing) |

| ResNet-101  | 95.37%              | config | [model](https://drive.google.com/file/d/1-_UcL3fDeU9jG0taKDB9Hv6gP5Bgg7Ns/view?usp=sharing) / [log](https://drive.google.com/file/d/1-0hrXmi3FGcxx5TmIwsb0m0xQBQrRupf/view?usp=sharing) |
| ResNet-152 | 95.18%              | config | [model](https://drive.google.com/file/d/1-jhTNhHOS-bqTXR33mvwRMa6aKnKhhU5/view?usp=sharing) / [log](https://drive.google.com/file/d/1-74tcVx1Avxz6FOU9IIYb9E7ZX3OmA5V/view?usp=sharing) |

## Citation

```bibtext
@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

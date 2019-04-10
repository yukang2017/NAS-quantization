# NAS-quantization
The code for [Joint Neural Architecture Search and Quantization](https://arxiv.org/abs/1811.09426)

## Requirements
* pytorch 0.4.1
* python 3.6
* [tensorboardX](https://github.com/lanpa/tensorboardX)

## Search
python train_search.py --data 'path_to_dataset'

## After search, CIFAR-10 training
For CIFAR-10 models, we rely on the traing code from [DARTS](https://github.com/quark0/darts) to train and evaluate the searched network.

## After search, ImageNet training
For ImageNet models, we rely on the traing code from [RENASNet](https://github.com/yukang2017/RENAS) to train and evaluate the searched network.

## Acknowledgements
Our implementation about quantization highly replies on the [quantized_distillation](https://github.com/antspy/quantized_distillation).
We also thanks for the inspiring work [DARTS](https://github.com/quark0/darts).

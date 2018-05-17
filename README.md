# chainer-Xception

This repository contains a [Chainer](https://chainer.org/) implementation for the paper: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (CVPR 2017, François Chollet).

## Requirements

- Chainer 4.0.0+
- CuPy 4.0.0+

## Use pretrained model

Download the pre-trained caffemodel from [Baidu Cloud](https://pan.baidu.com/s/1gfiTShd). 

If `pretrained_model` is specified as `xception.caffemodel`, it automatically loads and converts the caffemodel from `$CHAINER_DATASET_ROOT/pfnet/chainer/models/`, where `$CHAINER_DATASET_ROOT` os set as `$HOME/.chainer/dataset` unless you specify another value by modifying the environment variable.

``` python-console
Python 3.6.2 (default, Oct 31 2017, 12:23:24)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from xception import Xception

In [2]: model = Xception(pretrained_model='xception.caffemodel')
Now loading caffemodel (usually it may take few minutes)
```

## Reference

- [Chollet, François. "Xception: Deep Learning with Depthwise Separable Convolutions." arXiv preprint arXiv:1610.02357 (2016).](https://arxiv.org/abs/1610.02357)

import os

import chainer
import chainer.links as L
import chainer.functions as F

from chainer.initializers import normal


class SeparableConv2D(chainer.Chain):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 padding=0,
                 dilate=1,
                 initialW=None):

        super(SeparableConv2D, self).__init__()
        with self.init_scope():
            self.depthwise = L.Convolution2D(
                in_channels, in_channels, ksize, stride, padding,
                nobias=True, initialW=initialW, dilate=dilate, groups=in_channels)
            self.pointwise = L.Convolution2D(
                in_channels, out_channels, 1, 1, 0, dilate=1, groups=1,
                initialW=initialW, nobias=True)

    def __call__(self, x):
        h = self.depthwise(x)
        h = self.pointwise(h)

        return h


class Block(chainer.Chain):

    def __init__(self, in_filters, out_filters, initialW=None, start_with_relu=True):
        self.start_with_relu = start_with_relu

        super(Block, self).__init__()
        with self.init_scope():
            self.separable_conv = SeparableConv2D(
                in_filters, out_filters, 3, stride=1, padding=1, initialW=initialW)
            self.bn = L.BatchNormalization(out_filters)

    def __call__(self, x):
        if self.start_with_relu:
            x = F.relu(x)
        h = self.bn(self.separable_conv(x))

        return h


class BuildingBlock(chainer.Chain):

    def __init__(self, in_filters,
                 out_filters,
                 reps,
                 strides=1,
                 initialW=None,
                 start_with_relu=True,
                 grow_first=True):

        self.strides = strides
        self.grow_first = grow_first
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            if out_filters != in_filters or strides != 1:
                self.skip = L.Convolution2D(
                    in_filters, out_filters, 1, strides, initialW=initialW, nobias=True)
                self.skipbn = L.BatchNormalization(out_filters)
            else:
                self.skip = None

            self._forward = []
            filters = in_filters
            if grow_first:
                block_name = 'b_start'
                block = Block(in_filters, out_filters, initialW, start_with_relu)
                setattr(self, block_name, block)
                self._forward.append(block_name)
                filters = out_filters

            for i in range(1, reps):
                block_name = 'b{}'.format(i)
                block = Block(filters, filters, initialW)
                setattr(self, block_name, block)
                self._forward.append(block_name)

            if not grow_first:
                block_name = 'b_end'
                block = Block(in_filters, out_filters)
                setattr(self, block_name, block)
                self._forward.append(block_name)

    def __call__(self, x):
        h = x
        for name in self._forward:
            h = getattr(self, name)(h)

        if self.strides != 1:
            h = F.max_pooling_2d(h, 3, self.strides, 1, cover_all=False)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        return h + skip


class Xception(chainer.Chain):

    def __init__(self, pretrained_model):
        initialW = normal.HeNormal(scale=1.0)
        super(Xception, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 3, 2, 0, nobias=True, initialW=initialW)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(32, 64, 3, nobias=True, initialW=initialW)
            self.bn2 = L.BatchNormalization(64)

            self.block1 = BuildingBlock(
                64, 128, 2, 2, initialW, start_with_relu=False, grow_first=True)
            self.block2 = BuildingBlock(
                128, 256, 2, 2, initialW, start_with_relu=True, grow_first=True)
            self.block3 = BuildingBlock(
                256, 728, 2, 2, initialW, start_with_relu=True, grow_first=True)

            self.block4 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)
            self.block5 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)
            self.block6 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)
            self.block7 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)

            self.block8 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)
            self.block9 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)
            self.block10 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)
            self.block11 = BuildingBlock(
                728, 728, 3, 1, initialW, start_with_relu=True, grow_first=True)

            self.block12 = BuildingBlock(
                728, 1024, 2, 2, initialW, start_with_relu=True, grow_first=False)

            self.conv3 = SeparableConv2D(1024, 1536, 3, 1, 1, initialW=initialW)
            self.bn3 = L.BatchNormalization(1536)

            self.conv4 = SeparableConv2D(1536, 2048, 3, 1, 1, initialW=initialW)
            self.bn4 = L.BatchNormalization(2048)

            self.fc = L.Linear(2048, 1000)

        if pretrained_model and pretrained_model.endswith('.caffemodel'):
            _retrieve('xception.npz', pretrained_model, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None)
        _transfer_xception(caffemodel, chainermodel)
        chainer.serializers.save_npz(path_npz, chainermodel, compression=False)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = self.block9(h)
        h = self.block10(h)
        h = self.block11(h)
        h = self.block12(h)

        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        h = _global_average_pooling_2d(h)
        h = self.fc(h)

        return h


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))

    return h


def _transfer_xception(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.conv1_bn.avg_mean
    dst.bn1.avg_var[:] = src.conv1_bn.avg_var
    dst.bn1.gamma.data[:] = src.conv1_scale.W.data
    dst.bn1.beta.data[:] = src.conv1_scale.bias.b.data

    dst.conv2.W.data[:] = src.conv2.W.data
    dst.bn2.avg_mean[:] = src.conv2_bn.avg_mean
    dst.bn2.avg_var[:] = src.conv2_bn.avg_var
    dst.bn2.gamma.data[:] = src.conv2_scale.W.data
    dst.bn2.beta.data[:] = src.conv2_scale.bias.b.data

    for bnum in range(1, 13):
        _transfer_block(src, getattr(dst, 'block{}'.format(bnum)), bnum)

    _transfer_separable_conv(src.conv3_1, src.conv3_2, dst.conv3)
    _transfer_separable_conv(src.conv4_1, src.conv4_2, dst.conv4)

    dst.bn3.avg_mean[:] = src.conv3_bn.avg_mean
    dst.bn3.avg_var[:] = src.conv3_bn.avg_var
    dst.bn3.gamma.data[:] = src.conv3_scale.W.data
    dst.bn3.beta.data[:] = src.conv3_scale.bias.b.data

    dst.bn4.avg_mean[:] = src.conv4_bn.avg_mean
    dst.bn4.avg_var[:] = src.conv4_bn.avg_var
    dst.bn4.gamma.data[:] = src.conv4_scale.W.data
    dst.bn4.beta.data[:] = src.conv4_scale.bias.b.data

    dst.fc.W.data[:] = src.classifier.W.data
    dst.fc.b.data[:] = src.classifier.b.data


def _transfer_separable_conv(src_conv1, src_conv2, dst_conv):

    dst_conv.depthwise.W.data = src_conv1.W.data[src_conv1.W.data != 0].reshape(
        dst_conv.depthwise.W.data.shape)
    dst_conv.pointwise.W.data = src_conv2.W.data


def _transfer_match_conv(src, dst, prefix):
    match_conv_name = '{}_match_conv'.format(prefix)
    if hasattr(src, match_conv_name):
        match_conv = getattr(src, match_conv_name)
        match_conv_bn = getattr(src, '{}_bn'.format(match_conv_name))
        match_conv_scale = getattr(src, '{}_scale'.format(match_conv_name))

        dst.skip.W.data[:] = match_conv.W.data
        dst.skipbn.avg_mean[:] = match_conv_bn.avg_mean
        dst.skipbn.avg_var[:] = match_conv_bn.avg_var
        dst.skipbn.gamma.data[:] = match_conv_scale.W.data
        dst.skipbn.beta.data[:] = match_conv_scale.bias.b.data


def _transfer_block(src, dst, bnum):

    prefix = 'xception{}'.format(bnum)
    _transfer_match_conv(src, dst, prefix)

    for i, bname in enumerate(dst._forward):
        src_conv1 = getattr(src, '{}_conv{}_1'.format(prefix, i + 1))
        src_conv2 = getattr(src, '{}_conv{}_2'.format(prefix, i + 1))
        dst_block = getattr(dst, bname)
        src_block_prefix = '{}_conv{}'.format(prefix, i + 1)
        _transfer_components(src, dst_block, src_conv1, src_conv2, src_block_prefix)


def _transfer_components(src, dst, src_conv1, src_conv2, prefix):

    src_bn = getattr(src, '{}_bn'.format(prefix))
    src_scale = getattr(src, '{}_scale'.format(prefix))

    _transfer_separable_conv(src_conv1, src_conv2, dst.separable_conv)
    dst.bn.avg_mean[:] = src_bn.avg_mean
    dst.bn.avg_var[:] = src_bn.avg_var
    dst.bn.gamma.data[:] = src_scale.W.data
    dst.bn.beta.data[:] = src_scale.bias.b.data


def _make_npz(path_npz, path_caffemodel, model):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://pan.baidu.com/s/1gfiTShd#list/path=%2F\', '
            'and place it on {}'.format(path_caffemodel))

    Xception.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    chainer.serializers.load_npz(path_npz, model)
    return model


def _retrieve(name_npz, name_caffemodel, model):
    root = chainer.dataset.download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return chainer.dataset.download.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model),
        lambda path: chainer.serializers.load_npz(path, model))

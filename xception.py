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
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, ksize, stride, padding,
                nobias=True, initialW=initialW, dilate=dilate, group=in_channels)
            self.pointwise = L.Convolution2D(
                in_channels, out_channels, 1, 1, 0, dilate=1, group=1,
                initialW=initialW, nobias=True)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.pointwise(h)

        return h


class Block(chainer.Chain):

    def __init__(self, in_filters, out_filters, initialW=None, start_with_relu=True):
        self.start_with_relu = start_with_relu

        super(Block, self).__init__()
        with self.init_scope():
            self.separable_conv = SeparableConv2D(
                in_filters, out_filters, 3, stride=1, padding=1, initialw=initialW)
            self.bn = L.BatchNormalization(out_filters)

    def __call__(self, x):
        if self.start_with_relu:
            h = F.relu(x)
        h = self.bn(self.separable_conv(h))

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
            block = Block(filters, out_filters, initialW)
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
            l = getattr(self, name)
            h = l(h)

        if self.strides != 1:
            h = F.max_pooling_2d(h, 3, self.strides, 1)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        return h + skip


class Xception(chainer.Chain):

    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
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
                728, 1024, 3, 1, initialW, start_with_relu=True, grow_first=False)

            self.conv3 = SeparableConv2D(1024, 1536, 3, 1, 1, initialW=initialW)
            self.bn3 = L.BatchNormalization(1536)

            self.conv4 = SeparableConv2D(1536, 2048, 3, 1, 1, initialW=initialW)
            self.bn4 = L.BatchNormalization(2048)

            self.fc = L.Linear(2048, num_classes)

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

        h = F.average_pooling_2d(h, 1)
        h = self.fc(h)

        return h

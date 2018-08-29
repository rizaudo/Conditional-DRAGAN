import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image
import os
import argparse


class LayerMLP(chainer.Chain):
    def __init__(self, batchsize, n_hidden=100,  label_num=10, distribution='uniform', wscale=0.02):
        super(LayerMLP, self).__init__()
        self.n_hidden = n_hidden
        self.label_num = label_num
        self.distribution = distribution
        self.batchsize = batchsize
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(None, 256, initialW=w)
            self.l1 = L.Linear(256, 512, initialW=w)
            self.l2 = L.Linear(512, 256, initialW=w)
            self.l3 = L.Linear(256, 28 * 28, initialW=w)

    def make_input_z_with_label(self, batchsize: int, labelbatch: np.array):
        # labelbatch is 1d array
        # onehot representation
        xp = self.xp
        targets = labelbatch.reshape(-1)
        onehot = xp.eye(self.label_num)[targets]
        onehot = onehot.reshape(batchsize, self.label_num, 1, 1)

        if self.distribution == 'normal':
            nikome = xp.random.randn(batchsize, self.n_hidden - self.label_num, 1, 1)
            return xp.concatenate((onehot, nikome), axis=1).astype(np.float32)

        elif self.distribution == 'uniform':
            nikome = xp.random.uniform(-1, 1, (batchsize, self.n_hidden - self.label_num, 1, 1))
            return xp.concatenate((onehot, nikome), axis=1).astype(np.float32)

        else:
            raise ValueError('unknown z distribution: %s' % self.distribution)

    def __call__(self, x):
        self.x = F.relu(self.l0(x))
        self.x = F.relu(self.l1(self.x))
        self.x = F.relu(self.l2(self.x))
        self.x = F.tanh(self.l3(self.x))
        return self.x


class LayerMLPCritic(chainer.Chain):
    def __init__(self, wscale=0.02):
        super(LayerMLPCritic, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)

            self.l0 = L.Linear(None, 256, initialW=w)
            self.l1 = L.Linear(256, 512, initialW=w)
            self.l2 = L.Linear(512, 256, initialW=w)
            self.l3 = L.Linear(256, 1, initialW=w)

    def __call__(self, x):
        self.x = F.relu(self.l0(x))
        self.x = F.relu(self.l1(self.x))
        self.x = F.relu(self.l2(self.x))
        self.x = self.l3(self.x)
        return self.x


class LayerMLPClassifier(chainer.Chain):
    def __init__(self, label_num=10, wscale=0.02):
        super(LayerMLPClassifier, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(None, 256, initialW=w)
            self.l1 = L.Linear(256, 512, initialW=w)
            self.l2 = L.Linear(512, 256, initialW=w)
            self.l3 = L.Linear(256, label_num, initialW=w)

    def __call__(self, x):
        self.x = F.relu(self.l0(x))
        self.x = F.relu(self.l1(self.x))
        self.x = F.relu(self.l2(self.x))
        self.x = self.l3(self.x)
        return self.x


def make_label_one_image(label_num: int, rows: int, cols: int, gen, dst: str):
    n_images = rows * cols
    labels = np.ones(n_images, dtype=np.int8)
    z = Variable(gen.make_input_z_with_label(n_images, labels))
    x = gen(z)
    x = chainer.cuda.to_cpu(x.data)

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))

    preview_dir = '{}/vis'.format(dst)
    preview_path = preview_dir + '/image{:0>8}.png'.format(0)
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)
    return


def make_mnist_image(dst, gen, rows, cols, label_num=10, number=0):
    n_images = rows * cols
    a = np.empty(n_images, np.uint8)
    np.core.multiarray.copyto(a, number, casting='unsafe')
    labels = a
    xp = chainer.cuda.get_array_module(gen)
    z = Variable(gen.make_input_z_with_label(n_images, labels))
    x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    x = x.reshape((n_images, 1, 28, 28))
    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, ch, H, W = x.shape
    x = x.reshape((rows, cols, ch, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, ch))
    preview_dir = '{}/preview'.format(dst)
    preview_path = preview_dir + '/image{:0>8}.png'.format(number)
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    x = np.squeeze(x, axis=2)
    Image.fromarray(x, mode='L').save(preview_path)


def main():
    parser = argparse.ArgumentParser(description='GAN Visualizer')
    parser.add_argument('--rows', '-r', default=10)
    parser.add_argument('--cols', '-c', default=10)
    parser.add_argument('--gendir', '-dir', default=None)
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--fmnist', action='store_true')
    parser.add_argument('--allnum', action='store_true')
    args = parser.parse_args()

    if args.gendir is None:
        raise ValueError('generater-model is not selected!')
    if args.out is None:
        raise ValueError('out directory is not selected!')

    if not args.fmnist:
        raise NotImplementedError('this version is fmnist only...')
    else:
        generator = LayerMLP(batchsize=1, n_hidden=110, label_num=10)
    chainer.serializers.load_npz(args.gendir, generator)

    if not args.allnum:
        # Default: all one image.
        make_label_one_image(10, args.rows, args.cols, generator, dst=args.out)
    else:
        for i in range(10):
            make_mnist_image(dst=args.out, gen=generator, rows=args.rows, cols=args.cols, label_num=10, number=i)

    return


if __name__ == '__main__':
    main()

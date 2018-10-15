from __future__ import print_function
from PIL import Image
import os
import h5py
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.nnet import conv2d
from collections import OrderedDict
from argparse import ArgumentParser

"""
a random number generator used to initialize weights
"""
SEED = 123
rng = np.random.RandomState(SEED)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def opts_parser():

    usage = "LFNet Train"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, default=None, dest='path',
        help='Loading 4D training and validation LF from this path: (default: %(default)s)')
    parser.add_argument(
        '-N', '--sample_num', type=int, default=25, dest='sample_num',
        help='Number of Samples in use: (default: %(default)s)')
    parser.add_argument(
        '-L', '--length', type=int, default=9, dest='length',
        help='Length of 3D LF: (default: %(default)s)')
    parser.add_argument(
        '-F', '--factor', type=int, default=2, dest='factor',
        help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-B', '--batch_size', type=int, default=64, dest='batch_size',
        help='Batch Size: (default: %(default)s)')
    parser.add_argument(
        '-P', '--patch_size', type=int, default=48, dest='patch_size',
        help='Patch Size: (default: %(default)s)')
    parser.add_argument(
        '-M', '--load_model_flag', type=str2bool, default='False', dest='load_model_flag',
        help='Load Model Flag: (default: %(default)s)')

    return parser

class ConvLayer(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, filter_shape):
        """
        Allocate a c with shared variable internal parameters.

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type name: str
        :param name: given a special name for the ConvPoolLayer
        """

        # self.filter_shape = filter_shape
        # self.image_shape = image_shape
        # self.poolsize = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
        #            np.prod(poolsize))
        # initialize weights with random weights

        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.normal(0, 1e-3, size=filter_shape),
                dtype=config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(np.zeros(filter_shape[0]).astype(config.floatX), borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]

    def conv(self, input):
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            border_mode='half'
        )
        output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        return output

class LFNet(object):
    """LFNet Main Class"""

    def __init__(self, options):
        self.options = options

    def build_net(self, model):
        options = self.options
        # forward & backward data flow
        rn_x = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='rn_x')
        cn_x = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='cn_x')
        y = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='y')

        length = options['length']
        patch_size = options['patch_size']

        ## IMsF Layers
        IMsF_layer, IMsF_params = self._init_IMsF(options)


        ## Row Network
        rn_imsf_val = self._build_IMsF(rn_x, options, IMsF_layer, IMsF_params)

        # print(rn_imsf_val.get_value().shape)

        # BRCN-forward
        rn_layers_f, rn_params_f = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        rn_proj_f = self._build_model(rn_imsf_val, options, rn_layers_f, rn_params_f, go_backwards=False)

        # BRCN-backward
        rn_layers_b, rn_params_b = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        rn_proj_b = self._build_model(rn_imsf_val, options, rn_layers_b, rn_params_b, go_backwards=True)

        rn_brcn_params = dict(prefix_p('rn_f', rn_params_f), **(prefix_p('rn_b', rn_params_b)))

        ## Column Network
        cn_imsf_val = self._build_IMsF(cn_x,options,IMsF_layer,IMsF_params)

        # BRCN-forward
        cn_layers_f, cn_params_f = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        cn_proj_f = self._build_model(cn_imsf_val, options, cn_layers_f, cn_params_f, go_backwards=False)

        # BRCN-backward
        cn_layers_b, cn_params_b = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        cn_proj_b = self._build_model(cn_imsf_val, options, cn_layers_b, cn_params_b, go_backwards=True)

        cn_brcn_params = dict(prefix_p('cn_f', cn_params_f), **(prefix_p('cn_b', cn_params_b)))

        brcn_params = dict(rn_brcn_params, **cn_brcn_params)

        params = dict(IMsF_params, **brcn_params)

        stacked_w = self._init_stacked_weight(options['length'])

        # params = dict(net_params, **stacked_w)

        if model is not None:
            for k in params.iterkeys():
                params[k].set_value(model[k])

        proj = rn_proj_f.reshape((length,-1,length,patch_size,patch_size)).dimshuffle(1,0,2,3,4) \
               * stacked_w['rn_f_w'].dimshuffle('x',0,1,'x','x') \
               + rn_proj_b[::-1].reshape((length,-1,length,patch_size,patch_size)).dimshuffle(1,0,2,3,4) \
                 * stacked_w['rn_b_w'].dimshuffle('x',0,1,'x','x') \
               + cn_proj_f.reshape((length,-1,length,patch_size,patch_size)).dimshuffle(1,2,0,3,4) \
                 * stacked_w['cn_f_w'].dimshuffle('x',0,1,'x','x') \
               + cn_proj_b[::-1].reshape((length,-1,length,patch_size,patch_size)).dimshuffle(1,2,0,3,4) \
                 * stacked_w['cn_b_w'].dimshuffle('x',0,1,'x','x')

        weight_decay = theano.shared(numpy_floatX(0.), borrow=True)

        for v in params.itervalues():
            weight_decay += (v ** 2).sum()

        cost = tensor.mean(tensor.flatten((y-proj)**2)) + 1e-4 * weight_decay
        f_x = theano.function([rn_x, cn_x], proj, name='f_proj')

        return rn_x, cn_x, y, f_x, cost, params

    def _init_IMsF(self, options):
        layers = OrderedDict()
        params = OrderedDict()
        for i in range(len(options['IMsF_shape'])):
            # print('IMsF_'+str(i))
            layers['IMsF_' + str(i)] = ConvLayer(options['IMsF_shape'][i])
            params['IMsF_' + str(i) + '_w'] = layers['IMsF_' + str(i)].params[0]
            params['IMsF_' + str(i) + '_b'] = layers['IMsF_' + str(i)].params[1]

        return layers, params


    def _init_layer(self, filter_shape, rec_filter_size):
        """
        Global (net) parameter. For the convolution and regular opt.
        """
        layers = OrderedDict()
        params = OrderedDict()


        for i in range(len(filter_shape)):
            layers['conv_' + str(i) + '_v'] = ConvLayer(filter_shape[i])
            layers['conv_' + str(i) + '_t'] = ConvLayer(filter_shape[i])
            params['conv_' + str(i) + '_v_w'] = layers['conv_' + str(i) + '_v'].params[0]
            params['conv_' + str(i) + '_v_b'] = layers['conv_' + str(i) + '_v'].params[1]
            params['conv_' + str(i) + '_t_w'] = layers['conv_' + str(i) + '_t'].params[0]
            params['conv_' + str(i) + '_t_b'] = layers['conv_' + str(i) + '_t'].params[1]

            if i < len(rec_filter_size):
                layers['conv_' + str(i) + '_r'] = ConvLayer(rec_filter_size[i])
                params['conv_' + str(i) + '_r_w'] = layers['conv_' + str(i) + '_r'].params[0]
                params['conv_' + str(i) + '_r_b'] = layers['conv_' + str(i) + '_r'].params[1]

            params['b_' + str(i)] = theano.shared(np.zeros(filter_shape[i][0]).astype(config.floatX), name='b_' + str(i), borrow=True)


        return layers, params

    def _init_stacked_weight(self,length):

        stacked_w = OrderedDict()
        stacked_w['rn_f_w'] = theano.shared(np.ones((length,length)).astype(config.floatX)/4, borrow=True)
        stacked_w['rn_b_w'] = theano.shared(np.ones((length,length)).astype(config.floatX)/4, borrow=True)
        stacked_w['cn_f_w'] = theano.shared(np.ones((length,length)).astype(config.floatX)/4, borrow=True)
        stacked_w['cn_b_w'] = theano.shared(np.ones((length,length)).astype(config.floatX)/4, borrow=True)

        return stacked_w

    def _build_IMsF(self, input, options, layers, params):

        def _step(x_,layer_):
            layer_ = str(layer_.data)
            h_ = layers['IMsF_'+str(layer_)].conv(x_)

            return h_

        rval = input
        _rval = 0.0

        # rval0,_ = theano.scan(_step, sequences=[rval],
        #                           non_sequences=[0],
        #                           name='IMsF_layers_0')
        # rval1,_ = theano.scan(_step, sequences=[rval0],
        #                           non_sequences=[1],
        #                           name='IMsF_layers_1')
        # rval2,_ = theano.scan(_step, sequences=[rval1],
        #                           non_sequences=[2],
        #                           name='IMsF_layers_2')
        # rval3,_ = theano.scan(_step, sequences=[rval2],
        #                           non_sequences=[3],
        #                           name='IMsF_layers_3')



        for i in range(len(options['IMsF_shape'])):
            rval, _ = theano.scan(_step, sequences=[rval],
                                  non_sequences=[i],
                                  name='IMsF_layers_' + str(i))
            _rval += rval

        proj = tensor.nnet.relu(_rval)

        return proj



    def _build_model(self, input, options, layers, params, go_backwards=False):

        def _step1(x_, t_, layer_):
            layer_ = str(layer_.data)
            v = layers['conv_' + layer_ + '_v'].conv(x_)
            t = layers['conv_' + layer_ + '_t'].conv(t_)
            h = v + t

            return x_, h

        def _step2(h, r_, layer_):
            layer_ = str(layer_.data)
            o = h + params['b_' + layer_].dimshuffle('x', 0, 'x', 'x')
            if layer_ != str(len(options['filter_shape']) - 1):
                r = layers['conv_' + layer_ + '_r'].conv(r_)
                o = tensor.nnet.relu(o + r)
            return o

        rval = input
        if go_backwards:
            rval = rval[::-1]
        for i in range(len(options['filter_shape'])):
            rval, _ = theano.scan(_step1, sequences=[rval],
                                  outputs_info=[rval[0], None],
                                  non_sequences=[i],
                                  name='rnn_layers_k_' + str(i))
            rval = rval[1]
            rval, _ = theano.scan(_step2, sequences=[rval],
                                  outputs_info=[rval[-1]],
                                  non_sequences=[i],
                                  name='rnn_layers_q_' + str(i))
        # diff = options['padding']
        proj = rval \
               # + input

        return proj

def pred_error(f_pred,data_rn,data_cn,target):

    rn_x = data_rn
    cn_x = data_cn
    y = target
    pred = f_pred(rn_x,cn_x)

    pred = np.round(pred * 255.0)
    y = np.round(y * 255.0)

    z = np.mean((y - pred) ** 2)
    #
    # z /= x.shape[0] * x.shape[3] * x.shape[4]
    rmse = np.sqrt(z)
    # print('RMSE: ',rmse.eval())
    psnr = 20 * np.log10(255.0 / rmse)
    # psnr = tensor.sum(psnr)

    return psnr

def prefix_p(prefix, params):
    tp = OrderedDict()
    for kk, pp in params.items():
        tp['%s_%s' % (prefix, kk)] = params[kk]
    return tp

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return range(len(minibatches)), minibatches

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def load_model(path):
    npy = np.load(path)
    return npy.all()

def load_data(path):
    """the data is scalaed in [0 1]"""

    if not os.path.exists(path):
        raise IOError('No such File %s.' % path)

    if path.endswith('hdf5'):
        f = h5py.File(path,'r')
        lr_data = np.asarray(f.get('train_data')[:], dtype=config.floatX)
        hr_data = np.asarray(f.get('train_label')[:], dtype=config.floatX)
        v_lr_data = np.asarray(f.get('valid_data')[:], dtype=config.floatX)
        v_hr_data = np.asarray(f.get('valid_label')[:], dtype=config.floatX)

        print('Reading LF data from ', path)
        print('Train data Size', lr_data.shape, ' Range: ',lr_data.max(),lr_data.min())
        print('Train label Size', hr_data.shape, ' Range: ',hr_data.max(),hr_data.min())
        print('Validation data Size', v_lr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())
        print('Validation label size', v_hr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())

    elif path.endswith('mat'):
        f = h5py.File(path,'r')
        lr_data = np.asarray(f.get('train_data')[:], dtype=config.floatX).T
        hr_data = np.asarray(f.get('train_label')[:], dtype=config.floatX).T
        v_lr_data = np.asarray(f.get('valid_data')[:], dtype=config.floatX).T
        v_hr_data = np.asarray(f.get('valid_label')[:], dtype=config.floatX).T

        print('Reading LF data from ', path)
        print('Train data Size', lr_data.shape, ' Range: ',lr_data.max(),lr_data.min())
        print('Train label Size', hr_data.shape, ' Range: ',hr_data.max(),hr_data.min())
        print('Validation data Size', v_lr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())
        print('Validation label size', v_hr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())

    else:
        raise IOError('No .hdf5 or .mat file as input path.')


    return lr_data, hr_data, v_lr_data, v_hr_data

def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, rn_x, cn_x, y, cost, momentum=0):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([rn_x, cn_x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def train_LFNet(
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=1000,  # The maximum number of epoch to run
        dispFreq=1,  # Display to stdout the training progress every N updates
        lrate=1e-4,  # Learning rate for sgd (not used for adadelta and rmsprop)
        optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use,
        # not recommanded (probably need momentum and decaying learning rate).
        path = None,
        sample_num = 20,
        patch_size = 48,
        length = 7,
        factor = 3,
        validFreq=100,  # Compute the validation error after this number of update.
        saveFreq=200,  # Save the parameters after every saveFreq updates
        batch_size=64,  # The batch size during training and validateing.
        # Parameter for extra option
        momentum = 0,
        lmodel=True,  # Path to a saved model we want to start from.
):

    options = locals().copy()

    print('='*40)
    print('... Loading data')
    train_path = 'LFNet_train_EndToEnd_c1_s%d_l%d_f%d_p%d.mat' %(sample_num,length,factor,patch_size)
    options['train_path'] = train_path
    saveto='LFNet_trial_With_IMsF_f%d_l%d' %(factor,length)  # The best model will be saved there
    options['savetoModel'] = saveto
    model_path='LFNet_trial_With_IMsF_f%d_l%d_bak.npy' %(factor,length) # The model path
    train_set_x, train_set_y, valid_set_x, valid_set_y = load_data(train_path)

    options['sample_num'] = sample_num
    options['patch_size'] = patch_size
    options['factor'] = factor
    options['length'] = length
    options['batch_size'] = batch_size
    options['load_model_flag'] = lmodel

    options['IMsF_shape'] = [
        [64, 1, 3, 3],
        [64, 64, 3, 3],
        [64, 64, 3, 3],
        [64, 64, 3, 3]
    ]

    options['filter_shape'] = [
        [32, 64, 5, 5],
        [16, 32, 1, 1],
        [1, 16, 9, 9]
    ]
    options['rec_filter_size'] = [
        [32, 32, 1, 1],
        [16, 16, 1, 1]
    ]

    # options['padding'] = np.sum([(i[-1] - 1) / 2 for i in options['filter_shape']])
    print('='*40)
    print("model options", options)
    print('='*40)
    print('... Building model')

    net = LFNet(options)

    model = None
    if lmodel:
        model = load_model('./model/' + model_path)
    (rn_x, cn_x, y, f_x, cost, params) = net.build_net(model)

    f_cost = theano.function([rn_x, cn_x, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(params.values()))
    f_grad = theano.function([rn_x, cn_x, y], grads, name='f_grad')

    print('='*40)
    print('... Optimization')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, params, grads, rn_x, cn_x, y, cost)

    print('='*40)
    print('... Training')
    print("%d train examples" % ((train_set_x.shape[0] / batch_size) * batch_size))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = train_set_x.shape[0] // batch_size
    if saveFreq == -1:
        saveFreq = train_set_x.shape[0] // batch_size

    uidx = 0  # the number of update done

    try:
        for eidx in range(max_epochs):

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(train_set_x.shape[0], batch_size, shuffle=True)

            for train_index in kf[1]:
                uidx += 1
                # Select the random examples for this minibatch
                # diff = options['padding']
                x = np.asarray([train_set_x[t, :, :, :, :] for t in train_index])
                y = np.asarray([train_set_y[t, :, :, :, :] for t in train_index])

                rn_x = theano.shared(value=x, borrow=True).dimshuffle(1, 0, 2, 3, 4).reshape((length,-1,1,patch_size,patch_size)).eval()
                cn_x = theano.shared(value=x, borrow=True).dimshuffle(2, 0, 1, 3, 4).reshape((length,-1,1,patch_size,patch_size)).eval()
                y = theano.shared(value=y, borrow=True).eval()

                cost = f_grad_shared(rn_x, cn_x, y)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch {:03d} Update {:03d} Cost {}'.format(eidx,uidx,cost))

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = params
                    p = dict()
                    for k in params.iterkeys():
                        p[k] = np.asarray(params[k].eval()).astype(config.floatX)
                    np.save('./model/' + saveto, p)
                    print('Done')

                if np.mod(uidx, validFreq) == 0:

                    kv = get_minibatches_idx(valid_set_x.shape[0], batch_size, shuffle=False)

                    valid_psnr = []

                    for valid_index in kv[1]:
                        v_x = np.asarray([valid_set_x[v, :, :, :, :] for v in valid_index])
                        v_y = np.asarray([valid_set_y[v, :, :, :, :] for v in valid_index])

                        # v_rn_x = theano.shared(value=v_x, borrow=True).dimshuffle(1, 0, 2, 3, 4).reshape((length,-1,1,patch_size,patch_size)).eval()
                        # v_cn_x = theano.shared(value=v_x, borrow=True).dimshuffle(2, 0, 1, 3, 4).reshape((length,-1,1,patch_size,patch_size)).eval()

                        v_rn_x = v_x.transpose((1,0,2,3,4)).reshape((length,-1,1,patch_size,patch_size))
                        v_cn_x = v_x.transpose((2,0,1,3,4)).reshape((length,-1,1,patch_size,patch_size))

                        # v_y = theano.shared(value=v_y, borrow=True).eval()

                        valid_psnr.append(pred_error(f_x,v_rn_x,v_cn_x,v_y))

                    history_errs.append([np.average(valid_psnr)] + [cost])

                    if (best_p is None or
                                cost <= np.array(history_errs)[:, -1].min()):

                        best_p = params
                        bad_counter = 0
                    print('Epoch {:03d} Update {:03d} Validation PNSR {:.2f}dB'.format(eidx,uidx,np.average(valid_psnr)))

                    if (len(history_errs) > patience and
                                cost >= np.array(history_errs)[:-patience, -1].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            lrate /= 10.0
                            print('Downing learning rate for ', lrate, '\n')
                            bad_counter = 0


    except KeyboardInterrupt:
        print("Training interupted\n")


if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    path = args.path
    sample_num = args.sample_num
    length = args.length
    factor = args.factor
    patch_size = args.patch_size
    batch_size = args.batch_size
    load_model_flag = args.load_model_flag

    train_LFNet(max_epochs=30,sample_num=sample_num,length=length,factor=factor, lmodel=load_model_flag,
                patch_size=patch_size,batch_size=batch_size)

"""
LFNet_Test
Author: Yunlong Wang
Date: 2018.01
"""
from __future__ import print_function
import time
import os
import h5py
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
import skimage.io as io
import scipy.io as sio
from theano.tensor.nnet import conv2d
from collections import OrderedDict
from argparse import ArgumentParser
import gc
import datetime
import logging
import skimage
from skimage.transform import resize
from skimage import color
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()

"""
a random number generator used to initialize weights
"""
SEED = 123
rng = np.random.RandomState(SEED)


def opts_parser():

    usage = "LFNet Test"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, nargs='?', dest='path', metavar='PATH',
        help='Loading 4D training and validation LF from this path: (default: %(default)s)')
    parser.add_argument(
        '--scenes', type=str, nargs='+', metavar='SCENES', help='Namelist of LF scenes')
    parser.add_argument(
        '--model_path', type=str, nargs='?', metavar='MODEL_PATH',
        help='Loading pre-trained model file from this path: (default: %(default)s)')
    parser.add_argument(
        '--save_path', type=str, nargs='?', metavar='SAVE_PATH',
        help='Save Upsampled LF to this path: (default: %(default)s)')
    parser.add_argument(
        '-F', '--factor', type=int, default=4, metavar='FACTOR',
        choices=[2,3,4], help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-T', '--train_length', type=int, default=7, metavar='TRAIN_LENGTH',
        choices=[7,9], help='Training data length: (default: %(default)s)')
    parser.add_argument(
        '-C', '--crop_length', type=int, default=7, metavar='CROP_LENGTH',
        help='Crop Length from Initial LF: (default: %(default)s)')
    parser.add_argument(
        '-S', '--save_results', dest='save_results', action='store_true',
        help='Save Results or Not')

    return parser

class ConvLayer(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, filter_shape, std = 1e-3):
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
                rng.normal(0, std, size=filter_shape),
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
    """LFNet"""

    def __init__(self, options):
        self.options = options

    def build_net(self, model):
        options = self.options
        # forward & backward data flow
        x = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='x')

        # forward net
        IMsF_layer_f, IMsF_params_f = self._init_IMsF(options)
        x_f = self._build_IMsF(x, options, IMsF_layer_f, IMsF_params_f)
        brcn_layers_f, brcn_params_f = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        proj_f = self._build_model(x_f, options, brcn_layers_f, brcn_params_f, go_backwards=False)
        params_f = dict(IMsF_params_f, **brcn_params_f)

        # backward net
        IMsF_layer_b, IMsF_params_b = self._init_IMsF(options)
        x_b = self._build_IMsF(x, options, IMsF_layer_b, IMsF_params_b)
        brcn_layers_b, brcn_params_b = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        proj_b = self._build_model(x_b, options, brcn_layers_b, brcn_params_b, go_backwards=True)
        params_b = dict(IMsF_params_b, **brcn_params_b)

        params = dict(prefix_p('f', params_f), **(prefix_p('b', params_b)))

        if model is not None:
            for k in params.iterkeys():
                params[k].set_value(model[k])

        proj = (proj_f + proj_b[::-1])/2.0

        f_x = theano.function([x], proj, name='f_proj')

        return f_x


    def _init_IMsF(self, options):

        layers = OrderedDict()
        params = OrderedDict()

        IMsF_shape = options['IMsF_shape']

        for i in range(len(IMsF_shape)):
            # print('IMsF_'+str(i))
            layers['IMsF_' + str(i)] = ConvLayer(IMsF_shape[i],1e-1)
            params['IMsF_' + str(i) + '_w'] = layers['IMsF_' + str(i)].params[0]
            params['IMsF_' + str(i) + '_b'] = layers['IMsF_' + str(i)].params[1]
            params['IMsF_' + str(i) + '_rescale'] = theano.shared(np.ones(IMsF_shape[i][0]).astype(config.floatX),
                                                                  borrow=True)

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

    def _build_IMsF(self, input, options, layers, params):

        def _step(x_,layer_):
            layer_ = str(layer_.data)
            # print(layer_)
            h_ = layers['IMsF_'+str(layer_)].conv(x_)
            h_ = tensor.nnet.relu(h_)
            return h_

        rval = input
        _rval = 0.0

        for i in range(len(options['IMsF_shape'])):
            rval, _ = theano.scan(_step, sequences=[rval],
                                  non_sequences=[i],
                                  name='IMsF_layers_' + str(i))
            _rval += rval \
                     * params['IMsF_' + str(i) + '_rescale'].dimshuffle('x','x',0,'x','x')

        proj = _rval

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
               # + input[:,:,:,diff:-diff,diff:-diff]

        return proj

def pred_error(f_pred,data,target):

    x = data
    y = target
    pred = f_pred(x)

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

def load_model(path):
    npy = np.load(path)
    return npy.all()


def getSceneNameFromPath(path,ext):
    sceneNamelist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext):
                sceneName = os.path.splitext(name)[0]
                sceneNamelist.append(sceneName)

    sceneNamelist.sort()

def FolderTo4DLF(path,ext,length):
    path_str = path+'/*.'+ext
    log.info('-'*40)
    log.info('Loading %s files from %s' % (ext, path) )
    img_data = io.ImageCollection(path_str)
    if len(img_data)==0:
        raise IOError('No .%s file in this folder' % ext)
    # print(len(img_data))
    # print img_data[3].shape
    N = int(math.sqrt(len(img_data)))
    if not(N**2==len(img_data)):
        raise ValueError('This folder does not have n^2 images!')

    [height,width,channel] = img_data[0].shape
    lf_shape = (N,N,height,width,channel)
    log.info('Initial LF shape: '+str(lf_shape))
    border = (N-length)/2
    if border<0:
        raise ValueError('Border {0} < 0'.format(border))
    out_lf_shape = (height, width, channel, length, length)
    log.info('Output LF shape: '+str(out_lf_shape))
    lf = np.zeros(out_lf_shape).astype(config.floatX)
    # save_path = './DATA/train/001/Coll/'
    for i in range(border,N-border,1):
        for j in range(border,N-border,1):
            indx = j + i*N
            im = color.rgb2ycbcr(np.uint8(img_data[indx]))
            lf[:,:,0, i-border,j-border] = im[:,:,0]/255.0
            lf[:,:,1:3,i-border,j-border] = im[:,:,1:3]
            # io.imsave(save_path+str(indx)+'.png',img_data[indx])
    log.info('LF Range:')
    log.info('Channel 1 [%.2f %.2f]' %(lf[:,:,0,:,:].max(),lf[:,:,0,:,:].min()))
    log.info('Channel 2 [%.2f %.2f]' %(lf[:,:,1,:,:].max(),lf[:,:,1,:,:].min()))
    log.info('Channel 3 [%.2f %.2f]' %(lf[:,:,2,:,:].max(),lf[:,:,2,:,:].min()))
    log.info('--------------------')
    return lf


def AdjustTone(img,coef,norm_flag=False):

    log.info('--------------')
    log.info('Adjust Tone')

    tic = time.time()
    rgb = np.zeros(img.shape)
    img = np.clip(img,0.0,1.0)
    output = img ** (1/1.5)
    output = color.rgb2hsv(output)
    output[:,:,1] = output[:,:,1] * coef
    output = color.hsv2rgb(output)
    if norm_flag:
        r = output[:,:,0]
        g = output[:,:,1]
        b = output[:,:,2]
        rgb[:,:,0] = (r-r.min())/(r.max()-r.min())
        rgb[:,:,1] = (g-g.min())/(g.max()-g.min())
        rgb[:,:,2] = (b-b.min())/(b.max()-b.min())
    else:
        rgb = output

    log.info('IN Range: %.2f-%.2f' % (img.min(),img.max()))
    log.info('OUT Range: %.2f-%.2f' % (output.min(),output.max()))
    log.info("Elapsed time: %.2f sec" % (time.time() - tic))
    log.info('--------------')

    return  rgb

def modcrop(imgs,scale):

    if len(imgs.shape)==2:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row,:cropped_col]
    elif len(imgs.shape)==3:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row,:cropped_col,:]
    else:
        raise IOError('Img Channel > 3.')

    return  cropped_img

def ImgTo4DLF(filename,unum,vnum,length,adjust_tone,factor,save_sub_flag=False):

    if save_sub_flag:
        subaperture_path = os.path.splitext(filename)[0]+'_GT/'
        if not(os.path.exists(subaperture_path)):
            os.mkdir(subaperture_path)

    rgb_uint8 = io.imread(filename)
    rgb = np.asarray(skimage.img_as_float(rgb_uint8))
    log.info('Image Shape: %s' % str(rgb.shape))

    height = rgb.shape[0]/vnum
    width = rgb.shape[1]/unum
    channel = rgb.shape[2]

    if channel > 3:
        log.info('  Bands/Channels >3 Convert to RGB')
        rgb = rgb[:,:,0:3]
        channel = 3

    if adjust_tone > 0.0:
        rgb = AdjustTone(rgb,adjust_tone)

    cropped_height = height - height % factor
    cropped_width = width - width % factor

    lf_shape = (cropped_height, cropped_width, channel, vnum, unum)
    lf = np.zeros(lf_shape).astype(config.floatX)
    log.info('Initial LF shape: '+str(lf_shape))


    for i in range(vnum):
        for j in range(unum):
            im = rgb[i::vnum,j::unum,:]
            if save_sub_flag:
                subaperture_name = subaperture_path+'View_%d_%d.png' %(i+1,j+1)
                io.imsave(subaperture_name,im)
            lf[:,:,:,i,j] = color.rgb2ycbcr(modcrop(im,factor))
            lf[:,:,0,i,j] = lf[:,:,0,i,j]/255.0

    if unum % 2 == 0:
        border = (unum-length)/2 + 1
        u_start_indx = border
        u_stop_indx = unum - border + 1
        v_start_indx = border
        v_stop_indx = vnum - border + 1
    else:
        border = (unum-length)/2
        u_start_indx = border
        u_stop_indx = unum - border
        v_start_indx = border
        v_stop_indx = vnum - border

    if border<0:
        raise ValueError('Border {0} < 0'.format(border))

    out_lf = lf[:,:,:,v_start_indx:v_stop_indx,u_start_indx:u_stop_indx]
    log.info('Output LF shape: '+str(out_lf.shape))

    log.info('LF Range:')
    log.info('Channel 1 [%.2f %.2f]' %(out_lf[:,:,0,:,:].max(),out_lf[:,:,0,:,:].min()))
    log.info('Channel 2 [%.2f %.2f]' %(out_lf[:,:,1,:,:].max(),out_lf[:,:,1,:,:].min()))
    log.info('Channel 3 [%.2f %.2f]' %(out_lf[:,:,2,:,:].max(),out_lf[:,:,2,:,:].min()))
    log.info('--------------------')

    bic_lf = np.zeros(out_lf[:,:,0,:,:].shape).astype(config.floatX)

    for i in range(bic_lf.shape[2]):
        for j in range(bic_lf.shape[3]):
            this_im = out_lf[:,:,0,i,j]
            lr_im = resize(this_im, (cropped_height/factor,cropped_width/factor),
                           order=3, mode='symmetric', preserve_range=True)
            bic_lf[:,:,i,j] = resize(lr_im, (cropped_height,cropped_width),
                                     order=3, mode='symmetric', preserve_range=True)

    return out_lf, bic_lf

def del_files(path,ext):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext):
                os.remove(os.path.join(root, name))

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def test_LFNet(
        path = None,
        model_path = None,
        save_path = None,
        scene_names = None,
        train_length = 7,
        crop_length = 7,
        factor = 3,
        save_results = False
):

    options = locals().copy()

    if path is not None:
        log.info('='*40)
        if not os.path.exists(path):
            raise IOError('No such folder: {}'.format(path))
        if save_path is None:
            save_path = path+'_eval_l%d_f%d/'%(crop_length,factor)
        if not os.path.exists(save_path):
            log.warning('No such path for saving Our results, creating dir {}'
                        .format(save_path))
            mkdir_p(save_path)

        sceneNameTuple = tuple(scene_names)
        sceneNum = len(sceneNameTuple)

        if sceneNum == 0:
            raise IOError('No %s scene name in path %s' %(ext,eval_path))

    else:
        raise NameError('No folder given.')



    log_file = os.path.join(save_path,'LFNet_Test.log')
    if os.path.isfile(log_file):
        print('%s exists, delete it and rewrite...' % log_file)
        os.remove(log_file)
    fh = logging.FileHandler(log_file)
    log.addHandler(fh)

    log.info('='*40)
    log.info('Time Stamp: %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    total_PSNR = []
    total_SSIM = []
    total_Elapsedtime = []

    performacne_index_file = os.path.join(save_path,'performance_stat.mat')

    options['path'] = path
    options['Scenes'] = sceneNameTuple
    options['model_path'] = model_path
    options['save_path'] = save_path
    options['factor'] = factor
    options['train_length'] = train_length
    options['crop_length'] = crop_length
    options['save_results'] = save_results

    model_file = 'LFNet_RN_with_IMsF_f%d_l%d.npy' %(factor,train_length)

    if not os.path.exists(os.path.join(model_path,model_file)):
        raise IOError('No Such Model File %s', os.path.join(model_path,model_file))
    else:
        log.info('Loading pre-trained model from %s' % os.path.join(model_path,model_file))
        model = load_model(os.path.join(model_path,model_file))

    c_imsf = model['f_IMsF_0_w'].shape[0]
    c_in_imsf = model['f_IMsF_0_w'].shape[1]
    k_imsf = model['f_IMsF_0_w'].shape[-1]

    c1 = model['f_conv_0_v_w'].shape[0]
    c1_in = model['f_conv_0_v_w'].shape[1]
    k1 = model['f_conv_0_v_w'].shape[-1]
    c2 = model['f_conv_1_v_w'].shape[0]
    k2 = model['f_conv_1_v_w'].shape[-1]
    k3 = model['f_conv_2_v_w'].shape[-1]

    c0_r = model['f_conv_0_r_w'].shape[0]
    k0_r = model['f_conv_0_r_w'].shape[-1]
    c1_r = model['f_conv_1_r_w'].shape[0]
    k1_r = model['f_conv_1_r_w'].shape[-1]

    options['IMsF_shape'] = [
        [c_imsf, c_in_imsf, k_imsf, k_imsf],
        [c_imsf, c_imsf, k_imsf, k_imsf],
        [c_imsf, c_imsf, k_imsf, k_imsf],
        [c_imsf, c_imsf, k_imsf, k_imsf]
    ]

    options['filter_shape'] = [
        [c1, c1_in, k1, k1],
        [c2, c1, k2, k2],
        [c_in_imsf, c2, k3, k3]
    ]
    options['rec_filter_size'] = [
        [c0_r, c0_r, k0_r, k0_r],
        [c1_r, c1_r, k1_r, k1_r]
    ]

    # options['IMsF_shape'] = [
    #     [64, 1, 3, 3],
    #     [64, 64, 3, 3],
    #     [64, 64, 3, 3],
    #     [64, 64, 3, 3]
    # ]
    #
    # options['filter_shape'] = [
    #     [64, 64, 5, 5],
    #     [32, 64, 1, 1],
    #     [1, 32, 9, 9]
    # ]
    # options['rec_filter_size'] = [
    #     [64, 64, 1, 1],
    #     [32, 32, 1, 1]
    # ]

    # options['padding'] = np.sum([(i[-1] - 1) / 2 for i in options['filter_shape']])
    # diff = options['padding']

    log.info('='*40)
    log.info("model options\n"+str(options))

    log.info('='*40)
    tic = time.time()
    log.info('... Building pre-trained model' )
    net = LFNet(options)
    f_x = net.build_net(model)
    log.info("Elapsed time: %.2f sec" % (time.time() - tic))

    for scene in sceneNameTuple:
        log.info('='*15+scene+'='*15)
        if save_results:
            our_save_path = os.path.join(save_path,scene + '_OURS')
            GT_save_path = os.path.join(save_path,scene + '_GT')
            if os.path.isdir(our_save_path):
                log.info('='*40)
                del_files(our_save_path,'png')
                log.warning('Ours Save Path %s exists, delete all .png files' % our_save_path)
            else:
                os.mkdir(our_save_path)

            if os.path.isdir(GT_save_path):
                log.info('='*40)
                del_files(GT_save_path,'png')
                log.info('GT path %s exists, delete all .png files' % GT_save_path)
            else:
                os.mkdir(GT_save_path)

        if os.path.exists(os.path.join(path,scene+'.mat')):
            log.info('='*40)
            log.info('Loading GT and LR data from %s' % os.path.join(path,scene+'.mat'))
            dump = sio.loadmat(os.path.join(path,scene+'.mat'))
        else:
            raise IOError('No such .mat file: %s' % os.path.join(path,scene+'.mat'))

        lf = dump['gt_data'].astype(config.floatX)
        bic_lf = dump['lr_data'].astype(config.floatX)

        input_lf = lf[:,:,0,:,:]
        x_res = input_lf.shape[0]
        y_res = input_lf.shape[1]
        s_res = input_lf.shape[2]
        t_res = input_lf.shape[3]
        output_lf = np.zeros((x_res,y_res,s_res,t_res)).astype(config.floatX)

        log.info('='*40)
        s_time = time.time()
        log.info('LFNet SR running.....')
        log.info('>>>> Row Network')
        for s_n in range(s_res):
            row_seq = np.transpose(bic_lf[:,:,s_n,:],(2,0,1))
            up_row_seq = f_x(row_seq[:,np.newaxis,np.newaxis,:,:])
            output_lf[:,:,s_n,:] += np.transpose(up_row_seq[:,0,0,:,:],(1,2,0))
        log.info('>>>> Column Network')
        for t_n in range(t_res):
            col_seq = np.transpose(bic_lf[:,:,:,t_n],(2,0,1))
            up_col_seq = f_x(col_seq[:,np.newaxis,np.newaxis,:,:])
            output_lf[:,:,:,t_n] += np.transpose(up_col_seq[:,0,0,:,:],(1,2,0))
        output_lf /= 2.0
        process_time = time.time() - s_time
        log.info('Elapsed Time: %.2f sec per view'
                 % (process_time/(s_res*t_res)))

        PSNR = []
        SSIM = []

        log.info('='*40)
        log.info('Evaluation......')
        log.info('LR LF shape: %s' % str(bic_lf.shape))
        log.info('Predicted LF shape: %s' % str(output_lf.shape))
        log.info('GT LF shape: %s' % str(lf.shape))
        log.info('='*40)

        for s_n in xrange(s_res):
            for t_n in xrange(t_res):

                gt_img = lf[:,:,0,s_n,t_n]
                view_img = np.clip(output_lf[:,:,s_n,t_n],gt_img.min(),gt_img.max())
                bic_img = np.clip(bic_lf[:,:,s_n,t_n],gt_img.min(),gt_img.max())

                this_PSNR = psnr(np.uint8(view_img*255.0),np.uint8(gt_img*255.0))
                this_SSIM = ssim(np.uint8(view_img*255.0),np.uint8(gt_img*255.0))

                bic_PSNR = psnr(np.uint8(bic_img*255.0),np.uint8(gt_img*255.0))
                bic_SSIM = ssim(np.uint8(bic_img*255.0),np.uint8(gt_img*255.0))

                log.info('View %.2d_%.2d: PSNR: %.2fdB SSIM: %.4f' %(s_n+1, t_n+1, this_PSNR, this_SSIM))

                PSNR.append(this_PSNR)
                SSIM.append(this_SSIM)

                if save_results:
                    filename = os.path.join(our_save_path,'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    GTname = os.path.join(GT_save_path,'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    out_img = np.zeros((x_res,y_res,3))
                    gt_out_img = np.zeros((x_res,y_res,3))

                    out_img[:,:,0] = np.clip(view_img*255.0,16.0,235.0)
                    gt_out_img[:,:,0] = np.clip(gt_img*255.0,16.0,235.0)
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,0].max(),out_img[:,:,0].min()))
                    out_img[:,:,1:3] = lf[:,:,1:3,s_n,t_n]*255.0
                    gt_out_img[:,:,1:3] = lf[:,:,1:3,s_n,t_n]*255.0
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,1].max(),out_img[:,:,1].min()))

                    out_img = color.ycbcr2rgb(out_img)
                    out_img = np.clip(out_img,0.0,1.0)
                    out_img = np.uint8(out_img*255.0)

                    gt_out_img = color.ycbcr2rgb(gt_out_img)
                    gt_out_img = np.clip(gt_out_img,0.0,1.0)
                    gt_out_img = np.uint8(gt_out_img*255.0)

                    io.imsave(filename,out_img)
                    io.imsave(GTname,gt_out_img)


        log.info('='*40)
        total_PSNR.append(np.mean(np.array(PSNR)))
        total_SSIM.append(np.mean(np.array(SSIM)))
        total_Elapsedtime.append((process_time/(s_res*t_res)))
        log.info('[PSNR] Min: %.2f Avg: %.2f Max: %.2f dB' %(np.min(np.array(PSNR)),
                                                         np.mean(np.array(PSNR)),
                                                         np.max(np.array(PSNR))))
        log.info('[SSIM] Min: %.4f Avg: %.4f Max: %.4f' %(np.min(np.array(SSIM)),
                                                         np.mean(np.array(SSIM)),
                                                         np.max(np.array(SSIM))))
        log.info("[Elapsed time] %.2f sec per view." % (process_time/(s_res*t_res)))
        gc.collect()
        log.info('='*40)


    log.info('='*3+'Average Performance on %d scenes' % len(sceneNameTuple)+'='*6)
    log.info('PSNR: %.2f dB' % np.mean(np.array(total_PSNR)))
    log.info('SSIM: %.4f' % np.mean(np.array(total_SSIM)))
    log.info('Elapsed Time: %.2f sec per view' % np.mean(np.array(total_Elapsedtime)))
    log.info('='*40)

    embeded = dict(NAME=sceneNameTuple,PSNR=np.array(total_PSNR),SSIM=np.array(total_SSIM),TIME=np.array(total_Elapsedtime))
    sio.savemat(performacne_index_file,embeded)

if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    test_LFNet(path=args.path,model_path=args.model_path,factor=args.factor, train_length=args.train_length,
               crop_length=args.crop_length, scene_names=args.scenes, save_results=args.save_results)
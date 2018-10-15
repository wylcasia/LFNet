__author__ = 'Wang Yunlong'

import numpy as np
import os
from argparse import ArgumentParser
import skimage.io as io
import skimage
from skimage.transform import resize
from skimage import color
import time
import datetime
import scipy.io as sio
from scipy.misc import imresize
import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import compare_psnr as psnr

# plt.switch_backend('agg')
# matplotlib.use('Agg')

def opts_parser():
    usage = "Generate Training and Test Datasets for LFNet.\n" \
            "Error Code\n" \
            "10: This folder has Not n^2 images!\n" \
            "11: 'Length is larger than angularsize!'\n" \
            "12: 'Not RGB input!'\n" \
            "13: 'No such folder!'"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, default=None, dest='path',
        help='Loading 4D LF from this path: (default: %(default)s)')
    parser.add_argument(
        '-t', '--total_num', type=int, default=100, dest='total_num',
        help='Total numer of samples in this path: (default: %(default)s)')
    parser.add_argument(
        '-n', '--sample_num', type=int, default=25, dest='sample_num',
        help='Number of Samples in use: (default: %(default)s)')
    parser.add_argument(
        '-e', '--ext', type=str, default='png', dest='ext',
        help='Format of view images: (default: %(default)s)')
    parser.add_argument(
        '-l', '--length', type=int, default=9, dest='length',
        help='Length of 3D LF: (default: %(default)s)')
    parser.add_argument(
        '-f', '--factor', type=int, default=2, dest='factor',
        help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-p', '--patch_size', type=int, default=48, dest='patch_size',
        help='Patch Size: (default: %(default)s)')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=64, dest='batch_size',
        help='Batch Size: (default: %(default)s)')
    parser.add_argument(
        '-s', '--stride', type=int, default=36, dest='stride',
        help='Length of 3D LF: (default: %(default)s)')
    parser.add_argument(
        '-r', '--ratio', type=float, default=0.8, dest='ratio',
        help='Ratio for splitting train and test datasets: (default: %(default)s)')

    return parser

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


def generateMain(path, total_num, sample_num, ext, length, factor, patch_size, stride, ratio):

    data = []
    label = []

    sample_order = np.random.randint(total_num-1,size=sample_num)

    for ss in xrange(sample_num):
        sample_index = sample_order[ss] + 1
        path_str = path + '/%.3d/*.' % sample_index + ext
        print('--------------------')
        print('Loading %s files from %s' % (ext, path + '/%.3d' % sample_index) )
        img_data = io.ImageCollection(path_str)
        # print('Done.')
        # print(len(img_data))
        # print img_data[3].shape
        N = int(math.sqrt(len(img_data)))
        if not(N**2==len(img_data)):
            print('Exit. This folder has not n*n images')
            os._exit(10)

        [height,width,channel] = img_data[0].shape
        if not(channel==3):
            print('Exit. Not RGB input!')
            os._exit(12)

        lf_shape = (N,N,height,width,channel)
        print('LF shape: %s' % str(lf_shape))
        border = int((N-length)/2)
        if border<0:
            print('Exit. Length is larger than angularsize!')
            os._exit(11)

        cropped_height = height - height % factor
        cropped_width = width - width % factor
        all_gt = np.zeros((cropped_height,cropped_width,length)).astype(np.float32)
        all_lr = np.zeros((cropped_height,cropped_width,length)).astype(np.float32)

        print('Generating %d * %d patches' %(patch_size,patch_size))
        t0 = time.time()
        for i in range(border,N-border,1):
            for j in range(border,N-border,1):
                indx = j + i*N
                this_im = modcrop(color.rgb2ycbcr(img_data[indx])[:,:,0]/255.0,factor)
                all_gt[:,:,j-border] = this_im
                lr_im = resize(this_im, (cropped_height/factor,cropped_width/factor),
                               order=3, mode='symmetric', preserve_range=True)
                interp_im = resize(lr_im, (cropped_height,cropped_width),
                                              order=3, mode='symmetric', preserve_range=True)

                print('PSNR %.2f dB' % psnr(this_im,interp_im))
                all_lr[:,:,j-border] = interp_im

            for x in range(0,cropped_height-patch_size,stride):
                for y in range(0,cropped_height-patch_size,stride):
                    label.append(all_gt[x:x+patch_size,y:y+patch_size,:])
                    data.append(all_lr[x:x+patch_size,y:y+patch_size,:])

        print("Elapsed time: %.2f sec" % (time.time() - t0))


    label = np.asarray(label)
    data = np.asarray(data)

    print('='*20)
    sample_num = label.shape[0]
    print('Total Sample Number: %d' % sample_num)
    print('='*20)
    print('======Data Augmentation======')
    data = np.transpose(data,[0,3,1,2])
    label = np.transpose(label,[0,3,1,2])

    print('='*20)
    print('Shuffling Along First Axis')
    # order = np.random.permutation(sample_num)
    # data = data[order,:,:,:]
    # label = label[order,:,:,:]
    np.random.shuffle(data)
    np.random.shuffle(label)

    train_data = data[:int(ratio*sample_num)]
    train_label = label[:int(ratio*sample_num)]

    print('--------------------')
    print('Train DATA Size: '+str(train_data.shape))
    print('Train LABEL Size: '+str(train_label.shape))

    test_data = data[int(ratio*sample_num):]
    test_label = label[int(ratio*sample_num):]

    print('--------------------')
    print('Test DATA Size: '+str(test_data.shape))
    print('Test LABEL Size: '+str(test_label.shape))

    return train_data, train_label, test_data, test_label

if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    path = args.path
    total_num = args.total_num
    sample_num = args.sample_num
    ext = args.ext
    length = args.length
    factor = args.factor
    patch_size = args.patch_size
    stride = args.stride
    ratio = args.ratio

    global_channel = 1


    print('--------------------')
    print('Summary')
    print('Path: %s' % path)
    print('Total Num: %d' % total_num)
    print('Sample Num: %d' % sample_num)
    print('Format of Images: %s' % ext)
    print('Length: %d' % length)
    print('Upsampling Factor: %d' % factor)
    print('Patch Size: %d' % patch_size)
    print('Stride: %d' % stride)
    print('Ratio: %f' %ratio)

    if not(os.path.exists(path)):
        print('Exit. No such folder!')
        os._exit(13)

    t0 = time.time()
    [train_data, train_label, valid_data, valid_label] = generateMain(path=path, total_num=total_num,
                                                                      sample_num=sample_num, ext=ext, length=length,
                                                                      factor=factor, patch_size=patch_size,
                                                                      stride=stride, ratio=ratio)

    data_filename = 'LFNet_Train_c%d_s%d_l%d_f%d_p%d.hdf5' %(global_channel,sample_num,length,factor,patch_size)
    print('--------------------')
    print('Saving to %s file' % data_filename)

    f = h5py.File(data_filename,'w')
    train_group = f.create_group("Train")
    valid_group = f.create_group("Valid")
    label_chunksize = (64,length,patch_size,patch_size)
    data_chunksize = (64,length,patch_size,patch_size)
    train_data_des = f.create_dataset("train_data", data=train_data, chunks=data_chunksize, compression="gzip")
    train_label_des = f.create_dataset("train_label", data=train_label, chunks=label_chunksize, compression="gzip")
    valid_data_des = f.create_dataset("valid_data", data=valid_data, chunks=data_chunksize, compression="gzip")
    valid_label_des = f.create_dataset("valid_label", data=valid_label, chunks=label_chunksize, compression="gzip")
    train_group["data"] = train_data_des
    train_group["label"] = train_label_des
    valid_group["data"] = valid_data_des
    valid_group["label"] = valid_label_des
    print("Total Elapsed time: %.2f sec" % (time.time() - t0))




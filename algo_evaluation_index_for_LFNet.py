'''
Algorithm Comparisons --- Plotting Figures and Performance Index
'''
__author__ = 'Yunlong Wang'

import gc
import os
import os.path as op
import skimage.io as io
from skimage import color
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings
import math
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()

def opts_parser():
    usage = "Algorithm Evaluation, plotting figures and performance index."
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-R', '--root', type=str, default=None, dest='root',
        help='Root: (default: %(default)s)')
    parser.add_argument(
        '-S', '--scene', type=str, default=None, dest='scene_name',
        help='Scene: (default: %(default)s)')
    parser.add_argument(
        '-G', '--gt', type=str, default='GT', dest='GT',
        help='Ground Truth Folder: (default: %(default)s)')
    parser.add_argument(
        '--algo', type=str, default=None, dest='algo_name',
        help='Algorithm Name: (default: %(default)s)')
    parser.add_argument(
        '-E', '--ext', type=str, default='png', dest='ext',
        help='EXT: (default: %(default)s)')
    parser.add_argument(
        '-L', '--length', type=int, default=7, dest='length',
        help='Length to be dealt with: (default: %(default)s)')
    parser.add_argument(
        '--save_results', type=bool, default=True, dest='save_results',
        help='Save Results or Not: (default: %(default)s)')
    return parser

def remove_ticks_from_axes(axes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])


def FolderTo4DLF(path,ext,length):
    path_str = path+'/*.'+ext
    print('--------------------')
    print('[Ycbcr Single Channel] Loading %s files from %s' % (ext, path) )
    img_data = io.ImageCollection(path_str)
    if len(img_data)==0:
        print('No .%s file in this folder' % ext)
        os._exit(14)
    # print(len(img_data))
    # print img_data[3].shape
    N = int(math.sqrt(len(img_data)))
    if not(N**2==len(img_data)):
        print('This folder does not have n^2 images!')
        os._exit(13)
    [height,width,channel] = img_data[0].shape
    lf_shape = (N,N,height,width,channel)
    print('Initial LF shape: '+str(lf_shape))
    border = (N-length)/2
    if border<0:
        print('Length is larger than angularsize')
        os._exit(15)
    out_lf_shape = (length, length, height, width)
    print('Output LF shape: '+str(out_lf_shape))
    lf = np.zeros(out_lf_shape).astype(np.uint8)
    # save_path = './DATA/train/001/Coll/'
    for i in range(border,N-border,1):
        for j in range(border,N-border,1):
            indx = j + i*N
            im = color.rgb2ycbcr(np.uint8(img_data[indx]))
            lf[i-border,j-border,:,:] = im[:,:,0]
    print('LF Range: [%.2f %.2f]' %(lf.max(),lf.min()))
    print('--------------------')
    return lf

def FolderTo4DLF_RGB(path,ext,length):
    path_str = path+'/*.'+ext
    print('--------------------')
    print('[RGB mode] Loading %s files from %s' % (ext, path) )
    img_data = io.ImageCollection(path_str)
    if len(img_data)==0:
        print('No .%s file in this folder' % ext)
        os._exit(14)
    N = int(math.sqrt(len(img_data)))
    if not(N**2==len(img_data)):
        print('This folder does not have n^2 images!')
        os._exit(13)
    [height,width,channel] = img_data[0].shape
    lf_shape = (N,N,height,width,channel)
    print('Initial LF shape: '+str(lf_shape))
    border = (N-length)/2
    if border<0:
        print('Length is larger than angularsize')
        os._exit(15)
    out_lf_shape = (length, length, height, width, channel)
    print('Output LF shape: '+str(out_lf_shape))
    lf = np.zeros(out_lf_shape).astype(np.uint8)
    # save_path = './DATA/train/001/Coll/'
    for i in range(border,N-border,1):
        for j in range(border,N-border,1):
            indx = j + i*N
            # im = color.rgb2ycbcr(np.uint8(img_data[indx]))
            lf[i-border,j-border,:,:,:] = img_data[indx]
    print('LF Range C1: [%.2f %.2f]' %(lf[:,:,:,:,0].max(),lf[:,:,:,:,0].min()))
    print('LF Range C2: [%.2f %.2f]' %(lf[:,:,:,:,1].max(),lf[:,:,:,:,1].min()))
    print('LF Range C3: [%.2f %.2f]' %(lf[:,:,:,:,2].max(),lf[:,:,:,:,2].min()))
    print('--------------------')
    return lf


def compute_eval_index(lf_gt, lf_algo):


    assert lf_gt.shape[0:2] == lf_algo.shape[0:2]

    gt_height = lf_gt.shape[2]
    gt_width = lf_gt.shape[3]

    algo_height = lf_algo.shape[2]
    algo_width = lf_algo.shape[3]

    assert gt_height - algo_height >= 0
    assert gt_width - algo_width >= 0

    if (gt_height-algo_height) % 2 == 0:
        border_h = (gt_height-algo_height)/2
    else:
        log.warning('Border Not Divided Exactly.')
        border_h = (gt_height-algo_height)/2 + 1
        log.info('='*40)

    if (gt_width-algo_width) % 2 == 0:
        border_w = (gt_width-algo_width)/2
    else:
        log.warning('Border Not Divided Exactly.')
        border_w = (gt_width-algo_width)/2 + 1
        log.info('='*40)

    crop_lf_gt = lf_gt[:,:,border_h:border_h+algo_height,border_w:border_w+algo_width]

    PSNR = []
    SSIM = []

    log.info('='*40)
    for i in range(lf_gt.shape[0]):
        for j in range(lf_gt.shape[1]):
            gt_img = crop_lf_gt[i,j,:,:]
            algo_img = lf_algo[i,j,:,:]
            this_psnr = psnr(gt_img,algo_img)
            this_ssim = ssim(gt_img,algo_img)
            # res_line = 'View %.2d %.2d: PSNR %.2f dB SSIM %.4f' %(i+1,j+1,this_psnr,this_ssim)
            # log.info(res_line)
            PSNR.append(this_psnr)
            SSIM.append(this_ssim)
    log.info('='*40)

    log.info('PSNR min: %.2f  mean: %.2f max: %.2f dB' %(np.min(np.array(PSNR)),
                                                         np.mean(np.array(PSNR)),
                                                         np.max(np.array(PSNR))))
    log.info('SSIM min: %.4f  mean: %.4f max: %.4f' %(np.min(np.array(SSIM)),
                                                      np.mean(np.array(SSIM)),
                                                      np.max(np.array(SSIM))))
    log.info('='*40)

def MakeDir(path):

    if not op.isdir(path):
        os.mkdir(path)
        log.info('Creating Path: %s' % path)
    else:
        log.info('Path %s already exists.' % path)



def get_algorithm_lf(path, scene, algo_namelist, ext,length, lf_gt):

    lf_dict = dict()
    lf_dict_rgb = dict()

    for algo in algo_namelist:
        algo_folder = op.join(path, scene+'/'+algo)
        if not op.isdir(algo_folder):
            raise IOError('Scene %s Algorithm %s folder not found!' %(scene,algo_folder))
        else:
            lf_algo = FolderTo4DLF(algo_folder,ext,length)
            lf_dict[algo] = lf_algo
            lf_algo_rgb = FolderTo4DLF_RGB(algo_folder,ext,length)
            lf_dict_rgb[algo] = lf_algo_rgb

    # Crop each LF to the same Height & Width Size
    cropped_h = 0
    cropped_w = 0

    for algo in algo_namelist:
        lf_algo = lf_dict[algo]
        cur_h = lf_algo.shape[2]
        cur_w = lf_algo.shape[3]
        if cur_h < cropped_h and cur_w < cropped_w:
            cropped_h = cur_h
            cropped_w = cur_w

    print('-'*40)
    print('Cropped Height %d Width %d'%(cropped_h,cropped_w))
    print('-'*40)

    for algo in algo_namelist:
        lf_algo = lf_dict[algo]
        cur_h = lf_algo.shape[2]
        cur_w = lf_algo.shape[3]
        if cur_h > cropped_h > 0 and cur_w > cropped_w > 0:
            border_h = (cur_h - cropped_h)/2
            border_w = (cur_w - cropped_w)/2
            lf_dict[algo] = lf_algo[:,:,border_h:-border_h,border_w:-border_w]
            lf_dict_rgb[algo] = lf_algo_rgb[:,:,border_h:-border_h,border_w:-border_w,:]

    if cropped_h > 0 and cropped_w > 0:
        gt_h = lf_gt.shape[2]
        gt_w = lf_gt.shape[3]
        border_h = (gt_h - cropped_h)/2
        border_w = (gt_w - cropped_w)/2
        lf_gt_cropped = lf_gt[:,:,border_h:-border_h,border_w:-border_w,:]
    else:
        lf_gt_cropped = lf_gt

    return lf_dict,lf_dict_rgb,lf_gt_cropped

if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    root = args.root
    scene_name = args.scene_name
    GT = args.GT
    algo_name = args.algo_name
    ext = args.ext
    length = args.length

    log_file = os.path.join(root,'EVAL.log')
    if op.isfile(log_file):
        log.warning('%s exists, delete it and rewrite...' % log_file)
        os.remove(log_file)
    fh = logging.FileHandler(log_file)
    log.addHandler(fh)

    log.info('='*40)
    log.info('Summary')
    log.info('Date: %s' % datetime.datetime.now().strftime("%Y-%m-%d %H.%M"))
    log.info('Root Path: %s' %root)
    log.info('Scene Name List: %s' %scene_name)
    log.info('GT Pattern: %s' %GT)
    log.info('Alogrithm Name List: %s' %algo_name)
    log.info('Ext: %s' %ext)
    log.info('Length: %s' %length)
    log.info('='*40)

    if not op.isdir(root):
        raise IOError('No such folder: %s' %path)

    scene_list = scene_name.split(',')
    algo_list = algo_name.split(',')

    for scene in scene_list:

        log.info(' ')

        GT_folder = op.join(root,scene+'/'+GT)

        if not op.isdir(GT_folder):
            raise IOError('GT folder not found: %s' %GT_folder)
        else:
            lf_gt = FolderTo4DLF(GT_folder,ext,length)
            lf_gt_rgb = FolderTo4DLF_RGB(GT_folder,ext,length)

        algo_lf_dict,algo_lf_rgb,lf_gt_cropped = get_algorithm_lf(root,scene,algo_list,ext,length,lf_gt_rgb)

        log.info('='*20+scene+'='*20)
        for algo in algo_list:
            lf_algo = algo_lf_dict[algo]
            lf_algo_rgb = algo_lf_rgb[algo]
            log.info('-'*17+algo+'-'*17)
            compute_eval_index(lf_gt, lf_algo)












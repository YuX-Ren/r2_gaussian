import argparse
import os
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .mrc import *
# import utils

def add_args(parser):
    parser.add_argument(
        "volumes",
        type=os.path.abspath,
        help="Half-maps directory (.mrc)",
    )
    parser.add_argument(
        "--mask", metavar="mrc", type=os.path.abspath, help="FSC mask (.mrc)"
    )
    parser.add_argument(
        "--Apix", type=float, help="Angstroms per pixel"
    )
    return parser


def calc_fsc(vol1, vol2):
    """
    Helper function to calculate the FSC between two (assume masked) volumes
    vol1 and vol2 should be maps of the same box size, structured as numpy arrays with ndim=3, i.e. by loading with
    parse_mrc
    """
    # load masked volumes in fourier space

    vol1_ft = np.fft.fftshift(np.fft.fftn(vol1))
    vol2_ft = np.fft.fftshift(np.fft.fftn(vol2))

    # define fourier grid and label into shells
    Dx, Dy, Dz = vol1.shape
    x = np.arange(-Dx // 2, Dx // 2)
    y = np.arange(-Dy // 2, Dy // 2) 
    z = np.arange(-Dz // 2, Dz // 2)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    r_max = max(Dx, Dy, Dz) // 2  # sphere inscribed within volume box
    r_step = 1  # int(np.min(r[r>0]))
    bins = np.arange(0, r_max, r_step)
    bin_labels = np.searchsorted(bins, r, side="right")

    # calculate the FSC via labeled shells
    num = ndimage.sum(np.real(vol1_ft * np.conjugate(vol2_ft)), labels=bin_labels, index=bins + 1)
    den1 = ndimage.sum(np.abs(vol1_ft) ** 2, labels=bin_labels, index=bins + 1)
    den2 = ndimage.sum(np.abs(vol2_ft) ** 2, labels=bin_labels, index=bins + 1)
    fsc = num / np.sqrt(den1 * den2)

    x = bins / Dx  # x axis should be spatial frequency in 1/px
    return x, fsc

def fsc2res(freq, fsc, thresh=0.143, Apix=1):
    idx = np.searchsorted(-fsc, -thresh)
    if idx >= len(freq):
        res = 1 / freq[-1] * Apix
    else:
        res = 1 / freq[idx] * Apix
    return res

def analyze_fsc(gt_volume, pr_volume, Apix, save_dir):
    # dir1 = save_dir + '/halfA/'
    # dir2 = save_dir + '/halfB/'
    # files1 = []
    # for file in os.listdir(dir1):
    #     if file.endswith('.mrc'):
    #         files1.append(file)
    # files1.sort()
    # files2 = [] 
    # for file in os.listdir(dir2):
    #     if file.endswith('.mrc'):
    #         files2.append(file)
    # files2.sort()
    # assert len(files1) == len(files2), "Must have same number of maps in two halves"
    # if args.mask is not None:
    #     mask, _ = parse_mrc(args.mask, is_vol=True)
    # else:
    #     mask = None
    #     log('Warning: make sure the volumes unmasked')
    # Apix = get_voxelsize(dir1 + files1[0]).x
    # if np.isnan(Apix):
    #     Apix = 1.
    # if args.Apix is not None:
    #     Apix = args.Apix
    # fscs = []
    # count = 0
    # for file1, file2 in zip(files1, files2):
    #     count += 1
    #     log('Calculating GSFSC for {}th volume'.format(count))
    #     vol1, _ = parse_mrc(dir1 + file1, is_vol = True)
    #     vol2, _ = parse_mrc(dir2 + file2, is_vol = True)
    #     if mask is not None:
    #         freq, fsc = calc_fsc(vol1*mask, vol2*mask)
    #     else:
    #         freq, fsc = calc_fsc(vol1, vol2)
    #     fscs.append(fsc)
    # fscs = np.stack(fscs)
    # if mask is not None:
    #     vol_sum = (vol1+vol2)*mask/2
    # else:
    #     vol_sum = (vol1+vol2)/2
    # D = vol_sum.shape[0]
    freq, fsc = calc_fsc(gt_volume, pr_volume)
    print(fsc)
    res = fsc2res(freq, fsc, Apix=Apix)
    # utils.save_pkl(fscs, save_dir + '/fsc.pkl')
    plt.figure(1)
    plt.plot(freq, fsc)
    plt.axhline(0.143, c='k')
    plt.xlim([0, 0.5])
    plt.ylim([0, 1])
    plt.xticks(np.linspace(0, 0.5, 6), ['DC'] + ['{:.1f}'.format(1 / ele * Apix) for ele in np.linspace(0, 0.5, 6)[1:]])
    plt.xlabel('Frequency')
    plt.ylabel('GSFSC')
    plt.title('Resolution: {:.2f} A'.format(res))
    plt.savefig(save_dir + '/gsfsc.png', bbox_inches='tight')
    # plt.figure(2, figsize=(8,3))
    # plt.subplot(131)
    # plt.imshow(vol_sum[D//2].transpose())
    # plt.title('xslice')
    # plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(vol_sum[:, D//2].transpose())
    # plt.title('yslice')
    # plt.axis('off')
    # plt.subplot(133)
    # plt.imshow(vol_sum[:, :, D//2].transpose())
    # plt.title('zslice')
    # plt.axis('off')
    # plt.savefig(save_dir + '/volslice.png', bbox_inches='tight')
    # print('Save results to {}'.format(save_dir))
    # plt.show()

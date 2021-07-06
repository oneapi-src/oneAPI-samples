#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import colour
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004 as demosaic
import noise as noise

################################################################################
## Various parameters
## Changing these requires rerunning anr_gendata.py to generate input and output
## data for the FPGA test.
num_scene_captures = 2   # minimum 2

seed = 42
random_state = np.random.RandomState(seed)
exposure_time_multiplier = 1.0 / 8.0
dark_noise = 1.0
analog_gain = 1.0 / exposure_time_multiplier
digital_gain = 1.0

bits = 8
dynamic_range = 2**bits
black_level = 16
max_intensity = dynamic_range - 1
intensity_cutoff = (black_level, int(0.70 * max_intensity))
bin_neighborhood = 1

filter_size = 9
sigma_intensity_coef = 1.0
sigma_space = 4.0
alpha = 1.0
################################################################################

def write_parameter_config_file(filename, dark_noise, total_eff_gain):
  with open(filename, "w") as f:
    print('filter_size: {:d}'.format(filter_size), file=f)
    print('sig_shot: {}'.format(dark_noise), file=f)
    print('k: {}'.format(total_eff_gain), file=f)
    print('sig_i_coeff: {}'.format(sigma_intensity_coef), file=f)
    print('sig_s: {}'.format(sigma_space), file=f)
    print('alpha: {}'.format(alpha), file=f)

def write_bayer_data(img, filename):
  with open(filename, "w") as f:
    print('{} {}'.format(img.shape[0], img.shape[1]), file=f)
    for row in img:
      np.savetxt(f, row, fmt='%d', newline=' ')

def read_bayer_data(filename):
  with open(filename, "r") as f:
    lines = f.readlines()
    assert len(lines) == 2

    shape = tuple(map(int, lines[0].split(' ')))
    assert len(shape) == 2

    data = np.fromstring(lines[1], dtype=np.uint8, sep=' ')
    assert len(data) == (shape[0] * shape[1])

    return np.reshape(data, shape)


def rgb_to_bayer(img_ref):
  '''mosaicing'''
  w, h, _ = img_ref.shape
  img_bayer = np.zeros((2*w, 2*h, 3), dtype=np.uint8)
  img_bayer[::2, ::2, 2] = img_ref[:, :, 2]     # Blue (top-left)
  img_bayer[1::2, ::2, 1] = img_ref[:, :, 1]    # Green (top-right)
  img_bayer[::2, 1::2, 1] = img_ref[:, :, 1]    # Green (bottom-left)
  img_bayer[1::2, 1::2, 0] = img_ref[:, :, 0]   # Red (bottom-right)

  #Image.fromarray(img_bayer, "RGB").save("bayer_temp.png")

  img_bayer_gray = np.zeros((2*w, 2*h), dtype=np.uint8)
  img_bayer_gray[::2, ::2] = img_bayer[::2, ::2, 2]
  img_bayer_gray[1::2, ::2] = img_bayer[1::2, ::2, 1]
  img_bayer_gray[::2, 1::2] = img_bayer[::2, 1::2, 1]
  img_bayer_gray[1::2, 1::2] = img_bayer[1::2, 1::2, 0]

  return img_bayer_gray


def bayer_to_rgb(img_ref):
  '''demosaicing'''
  img_raw_pad = np.pad(img_ref, 2)
  img_raw_pad[  :2,  : ] = img_raw_pad[ 2: 4,  :  ]
  img_raw_pad[-2: ,  : ] = img_raw_pad[-4:-2,  :  ]
  img_raw_pad[  : ,  :2] = img_raw_pad[  :  , 2: 4]
  img_raw_pad[  : ,-2: ] = img_raw_pad[  :  ,-4:-2]
  img_rgb = demosaic(img_raw_pad, 'BGGR')[2:-2,2:-2] / 2**(bits - 8)
  return noise.saturate(img_rgb, 0, 255).astype(np.uint8)

def add_and_estimate_noise(img_ref):
  # capture two dark (black) images
  dark = []
  for i in range(num_scene_captures):
    dark.append(
        noise.noiser(
            np.zeros_like(img_ref),
            exposure_time_multiplier=exposure_time_multiplier,
            dark_noise=dark_noise, 
            analog_gain=analog_gain, 
            digital_gain=digital_gain, 
            bits=bits,
            black_level=black_level,
            saturate_return=True,
            random_state=random_state))

  # capture minimum two images from a perfectly still scene with a wide and balanced histogram
  scene = []
  for i in range(num_scene_captures):
    scene.append(
        noise.noiser(
            img_ref, 
            exposure_time_multiplier=exposure_time_multiplier,
            dark_noise=dark_noise, 
            analog_gain=analog_gain, 
            digital_gain=digital_gain, 
            bits=bits,
            black_level=black_level,
            saturate_return=True,
            random_state=random_state))

  for j in range(num_scene_captures-1):
    # estimate noise parameters
    noise_params, black_level_est, noise_curve = noise.noise_estimator(
        dark[j], scene[j:j+2], bits, intensity_cutoff, bin_neighborhood)

  return scene[-1], noise_params[0], noise_params[1]



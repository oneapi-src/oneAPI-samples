import anr_util as anr_util
import numpy as np
from PIL import Image
import noise as noise
import sys
import os

# parse command line args
img_file = '../images/reilly1.jpg'
out_dir = '../test_data'
if len(sys.argv) > 1:
  img_file = sys.argv[1]
if len(sys.argv) > 2:
  out_dir = sys.argv[2]

# parse input image
img_ref = np.array(Image.open(img_file))
w, h, _ = img_ref.shape

# filenames use the input base filename
filename_no_ext = os.path.splitext(os.path.basename(img_file))[0]
img_in_filename = out_dir + "/" + filename_no_ext + "_input.png"
img_in_noisy_filename = out_dir + "/" + filename_no_ext + "_input_noisy.png"
img_in_noisy_data_filename = out_dir + "/" + filename_no_ext + "_input_noisy.data"
param_config_data_filename = out_dir + "/" + filename_no_ext + "_param_config.data"
img_out_ref_data_filename = out_dir + "/" + filename_no_ext + "_output_ref.data"
img_out_ref_filename = out_dir + "/" + filename_no_ext + "_output_ref.png"

################################################################################
print('Converting RGB->Bayer->RGB')
# Convert from RGB->Bayer->RGB by mosaicing and then demosaicing
img_ref_rgb_bayer_rgb = anr_util.bayer_to_rgb(anr_util.rgb_to_bayer(img_ref))

# write out RGB image after
Image.fromarray(img_ref_rgb_bayer_rgb, "RGB").save(img_in_filename)
################################################################################

################################################################################
print('Adding noise to the image')
# Add some noise to the image after going RGB->Bayer->RGB
img_ref_noisy, dark_noise_estimate_with_gain, total_effective_gain_estimate = anr_util.add_and_estimate_noise(img_ref_rgb_bayer_rgb)
img_ref_noisy_uint = noise.saturate(img_ref_noisy, 0, 2**anr_util.bits-1).astype(np.uint8)

# write out the RGB image after adding noise
Image.fromarray(img_ref_noisy_uint).save(img_in_noisy_filename, bits=24)

# convert noisy image to Bayer format
img_ref_noisy_bayer = anr_util.rgb_to_bayer(img_ref_noisy_uint)

# write out the raw Bayer data
anr_util.write_bayer_data(img_ref_noisy_bayer, img_in_noisy_data_filename)

# write configuration file
anr_util.write_parameter_config_file(param_config_data_filename, dark_noise_estimate_with_gain, total_effective_gain_estimate)
################################################################################

################################################################################
print('Denoising the image')
# Denoise the image
r = anr_util.filter_size//2
img_ref_noisy_bayer_padded = np.pad(img_ref_noisy_bayer, r)
sigma_intensity = noise.sqrt(np.abs(img_ref_noisy_bayer_padded), dark_noise_estimate_with_gain, total_effective_gain_estimate)
sigma_intensity *= anr_util.sigma_intensity_coef
img_filtered = noise.bilateral_filter_raw(img_ref_noisy_bayer_padded, anr_util.filter_size, sigma_intensity, anr_util.sigma_space)[r:-r,r:-r]
img_filtered_alpha = anr_util.alpha * img_filtered + (1.0 - anr_util.alpha) * img_ref_noisy_bayer

# write out the raw Bayer data
anr_util.write_bayer_data(img_filtered_alpha, img_out_ref_data_filename)

# convert to RGB and write out reference output image
img_filtered_rgb = anr_util.bayer_to_rgb(img_filtered_alpha)
Image.fromarray(img_filtered_rgb).save(img_out_ref_filename, bits=24)
################################################################################

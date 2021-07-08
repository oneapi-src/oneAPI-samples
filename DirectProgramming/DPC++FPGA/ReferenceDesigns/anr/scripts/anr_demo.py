import anr_util as anr_util
import numpy as np
from PIL import Image
import noise as noise
import sys
import os

# parse command line args
fpga_exe = "anr.fpga"
img_file = '../images/reilly1.jpg'
out_dir = '../test_data'
if len(sys.argv) > 1:
  fpga_exe = sys.argv[1]
if len(sys.argv) > 2:
  img_file = sys.argv[2]
if len(sys.argv) > 3:
  out_dir = sys.argv[3]

# create output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# parse input image
img_ref = np.array(Image.open(img_file))
w, h, _ = img_ref.shape

# filenames use the input base filename
filename_no_ext = os.path.splitext(os.path.basename(img_file))[0]
img_in_filename = out_dir + "/" + filename_no_ext + "_input.png"
img_in_noisy_filename = out_dir + "/input_noisy.png"
img_in_noisy_data_filename = out_dir + "/input_noisy.data"
param_config_data_filename = out_dir + "/param_config.data"
img_out_ref_data_filename = out_dir + "/output_ref.data"
img_out_ref_filename = out_dir + "/output_ref.png"
fpga_out_data_filename = out_dir + "/output.data"
fpga_out_img_filename = out_dir + "/output.png"

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
print('Running FPGA kernel')
fpga_exe_dir = os.path.dirname(fpga_exe)
fpga_exe_file = os.path.basename(fpga_exe)

# save cwd
curr_cwd=os.getcwd()

# move to the directory of the exe
print('Moving directory: cd {}'.format(fpga_exe_dir))
os.system('cd {}'.format(fpga_exe_dir))

# run the exe
fpga_runs = 2
fpga_batches = 1 
command='./{} {} {} {}'.format(fpga_exe_file, os.path.abspath(out_dir), fpga_runs, fpga_batches)
os.system(command)

# move back to cwd from before running the fpga exe
print('Moving directory: cd {}'.format(curr_cwd))
os.system('cd {}'.format(curr_cwd))

# get results from the FPGA
img_fpga_bayer = anr_util.read_bayer_data(fpga_out_data_filename)

# convert Bayer to RGB
img_fpga_rgb = anr_util.bayer_to_rgb(img_fpga_bayer)

# print results
Image.fromarray(img_fpga_rgb).save(fpga_out_img_filename, bits=24)
################################################################################

################################################################################
# TODO: display output nicely
################################################################################
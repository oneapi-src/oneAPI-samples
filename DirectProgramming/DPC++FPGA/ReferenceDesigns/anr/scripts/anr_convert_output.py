import anr_util as anr_util
import numpy as np
from PIL import Image
import noise as noise
import sys
import os

# parse command line args
if len(sys.argv) <= 1:
  print('ERROR: not enough args')
else:
  img_file = sys.argv[1]
  out_dir = os.path.dirname(img_file)
  filename_no_ext = os.path.splitext(os.path.basename(img_file))[0]

  bayer_data = anr_util.read_bayer_data(img_file)
  rgb_data = anr_util.bayer_to_rgb(bayer_data)

  out_file_path = out_dir + "/" + filename_no_ext + ".png"
  print('Writing output file to {}'.format(out_file_path))
  Image.fromarray(rgb_data, "RGB").save(out_file_path)

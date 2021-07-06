import anr_util as anr_util
import numpy as np
from PIL import Image
import noise as noise
import sys

# parse command line args
img_file = '../data/reilly1.jpg'
if len(sys.argv) > 1:
  img_file = sys.argv[1]

# parse input image
img_ref = np.array(Image.open(img_file))
w, h, _ = img_ref.shape

# 1) TODO: Parse RGB
# 2) TODO: Add noise
# 3) TODO: RGB->Bayer (mosaicing); dump to file
# 4) TODO: Run oneAPI program passing file from (3)
# 5) TODO: Read output from oneAPI program
# 6) TODO: Bayer->RGB (demosaicing)
# 7) TODO: Display output
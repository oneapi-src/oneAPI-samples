## Datasets
This is a small neutrinos clustering app.  It supports several modes of clustering, kmeans, dbscan, heirarchical.  One of these modes (heirarchical) uses a knn graph.

Requirements:
 - h5py
 - sklearn
 - numpy
 - matplotlib (for optional visualization output)

Everything else is core python.

There are 4 files in this folder that are useful for this clustering app.  Each file reads in an image that is approximately 1000x2000 pixels, and there are 3 images per "event" (3 distinct views of a 3D object).  There are approximately 15000 images per file.

This script will loop over the files, read in images, find the non-zero locations, and perform clustering on them.

To run the script, use `python cluster_images.py --input file1.h5 [file2.h5 ...] --method [dbscan | kmeans | heirarchical] [-n number-of-events-to-process] [--visualize]`.

The script uses argparse;  run `python cluster_images.py --help`

For questions, please contact Corey Adams: corey.adams@anl.gov

To run the workload on different devices, use modified `cluster_images_intel.py` script with support of Intel CPUs and GPUs.
Run `python cluster_images_intel.py --help` for more info.



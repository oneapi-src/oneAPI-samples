import numpy as np
import colour
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004 as demosaic
from scipy.optimize import curve_fit

# Saturates the input within the range [low,high]
def saturate(x, low, high):
    x[x > high] = high
    x[x < low] = low
    return x

# Artificial temporal noise generator
def noiser(
    img_ref,
    exposure_time_multiplier=1.0, # a factor of exposure time (not actual time in seconds)
    quantum_efficiency=1.0,       # quantum efficiency of the sensor
    dark_noise=0.0,               # base dark noise parameter of the sensor
    analog_gain=1.0,              # analog gain factor (not real gain, but a scaler)
    digital_gain=1.0,             # digital gain factor
    bits=8,                       # number of quantization bits
    black_level=0,                # black level pedestal
    saturate_return=True,         # saturate the noisy image before returning
    random_state=np.random.RandomState(seed=42)):

    # Shot noise: part of the photon capture process that is a poisson distribution
    img = random_state.poisson(img_ref.astype(np.float) * np.float(exposure_time_multiplier))
    # Quantum efficiency: photon to electron conversion probability
    img = img * quantum_efficiency
    # Dark noise: false electrons generated (or lost) as a thermal process
    img += random_state.normal(scale=dark_noise, size=img.shape)
    # Gains: analog and digital gains with quantization
    img = ((img * np.float(analog_gain)).astype(np.int) * digital_gain).astype(np.int)
    # Black level: offset ensuring that negative part of the dark noise is retained
    img += black_level
    # Saturation
    if saturate_return:
        saturate(img, 0, int(2 ** bits - 1))
    
    return img

# Temporal noise function
def sqrt(x, dark_noise, k):
    return np.sqrt(k * x + dark_noise**2)

# Estimates temporal noise function from a dark frame and two frames from the same static scene
def noise_estimator(
    dark_frame,            # a dark frame captured with the same settings as the scene
    scene_frames,          # a couple of frames from a static scene with a wide histogram
    bits,                  # number of quantization bits
    intensity_cutoff=None, # reliable intensity range to be used in curve fitting
    bin_neighborhood=3     # the number of neighboring intensities to bin
    ):

    # if a tuple is given then skip default values
    if intensity_cutoff is None:
        # default range is 0 to 70% of the dynamic range
        intensity_cutoff = (0, 0.7 * 2**bits-1)
    elif np.isscalar(intensity_cutoff):
        # if a scalar is provided it is the upper range
        intensity_cutoff = (0, intensity_cutoff)

    # dark noise and black level are estimated from the dark frame
    dark_noise_est = np.std(dark_frame)
    black_level_est = np.int(np.round(np.mean(dark_frame)))

    # average of the two scene frames is used as a proxy for the expected value
    scene_avg = (scene_frames[0] + scene_frames[1]) / 2
    # first element of the noise curve is the dark noise estimated from the dark captures accurately
    noise_curve = np.zeros(2**bits)
    noise_curve[black_level_est] = dark_noise_est
    for i in range(black_level_est+1, 2**bits):
        # Bin a small range of intensities to avoid scene histogram inequality noise
        bin_mask = np.logical_and(scene_avg <= i+bin_neighborhood, scene_avg >= i-bin_neighborhood)
        if np.count_nonzero(bin_mask) > 50:
            # calculate standard deviations per bin
            noise_curve[i] = np.std(scene_frames[0][bin_mask] - scene_frames[1][bin_mask]) / np.sqrt(2)
        else:
            # too few samples, set the bin to NaN to indicate
            noise_curve[i] = np.NaN

    # NaN values are meant to be zero
    noise_curve = np.nan_to_num(noise_curve)

    # fit a square root curve to the noise estimations
    x_data = np.arange(0, intensity_cutoff[1] - intensity_cutoff[0])
    y_data = noise_curve[intensity_cutoff[0]:intensity_cutoff[1]]
    # give equal weight to all noise sample points, except for the first point (dark noise)
    # which is known to be more accurate
    y_sigma = np.ones_like(y_data)
    y_sigma[0] = 0.25
    # fit the curve
    noise_params, _ = curve_fit(
        f=sqrt, xdata=x_data, ydata=y_data, sigma=y_sigma, p0=(dark_noise_est, 1.0),
        bounds=((0.0, 0.0), (np.inf, np.inf)))

    return noise_params, black_level_est, noise_curve

# Gaussian function
def gaussian(src, sigma):
    dst = np.exp(-1/2 * (src / sigma)**2)
    return dst

# Calculate weight kernel of a bilateral filter
def calc_kernel(src, sigmaColor, sigmaSpace):
    k = src.shape[0]
    r = k//2
    center = src[r]
    intensity_kernel = gaussian(np.abs(center-src), sigmaColor)

    grid = np.mgrid[-r:r+1]
    spatial_kernel = gaussian(grid, sigmaSpace)
    if len(src.shape) == 2:
        spatial_kernel = spatial_kernel[:, None]
        spatial_kernel = np.tile(spatial_kernel, (1, 3))

    return intensity_kernel * spatial_kernel

# RGB bilateral filter
def bilateral_filter(src, d, sigmaColor, sigmaSpace, borderType=None, is_seperable=True):
    dst = np.zeros_like(src, dtype=np.float32)
    tmp = np.zeros_like(src, dtype=np.float32)
    r = d//2


    if is_seperable:
        for i in range(r, src.shape[0]-r):
            for j in range(0, src.shape[1]):
                src_local = src[i-r:i+r+1, j, :].astype(np.float32)
                kernel = calc_kernel(src_local, sigmaColor[i,j,:], sigmaSpace)
                tmp[i, j] = (src_local * kernel).sum(axis=0) / kernel.sum(axis=0)
        for i in range(r, src.shape[0]-r):
            for j in range(r, src.shape[1]-r):
                src_local = tmp[i, j-r:j+r+1, :]
                kernel = calc_kernel(src_local, sigmaColor[i,j,:], sigmaSpace)
                dst[i,j] = (src_local * kernel).sum(axis=0) / kernel.sum(axis=0)
    else:
        for i in range(r, src.shape[0]-r):
            for j in range(r, src.shape[1]-r):
                src_local = src[i-r:i+r+1, j-r:j+r+1, :].astype(np.float32)
                kernel = calc_kernel(src_local, sigmaColor[i,j,:], sigmaSpace)
                dst[i,j] = (src_local * kernel).sum(axis=(0,1)) / kernel.sum(axis=(0,1))

    return dst

# RAW bilateral filter
def bilateral_filter_raw(src, d, sigmaColor, sigmaSpace, borderType=None, is_seperable=True):
    dst = np.zeros_like(src, dtype=np.float32)
    tmp = np.zeros_like(src, dtype=np.float32)
    r = d//2

    if is_seperable:
        for i in range(r, src.shape[0]-r):
            for j in range(0, src.shape[1]):
                src_local = src[i-r:i+r+1:2, j].astype(np.float32)
                kernel = calc_kernel(src_local, sigmaColor[i,j], sigmaSpace,)
                tmp[i, j] = (src_local * kernel).sum(axis=0) / kernel.sum(axis=0)
        for i in range(r, src.shape[0]-r):
            for j in range(r, src.shape[1]-r):
                src_local = tmp[i, j-r:j+r+1:2]
                kernel = calc_kernel(src_local, sigmaColor[i,j], sigmaSpace)
                dst[i,j] = (src_local * kernel).sum(axis=0) / kernel.sum(axis=0)
    else:
        for i in range(r, src.shape[0]-r):
            for j in range(r, src.shape[1]-r):
                src_local = src[i-r:i+r+1, j-r:j+r+1, :].astype(np.float32)
                kernel = calc_kernel(src_local, sigmaColor[i,j,:], sigmaSpace)
                dst[i,j] = (src_local * kernel).sum(axis=(0,1)) / kernel.sum(axis=(0,1))

    return dst

# Simple ISP
def isp(img_raw, black_level, bits, ab_weights, ccm_matrix, gamma, rggb):
    # Black level removal and white balance
    img_raw_wb = ((img_raw - black_level) / (2**bits-1)) * (2**bits-1)
    img_raw_wb[ ::2,  ::2] = img_raw_wb[ ::2,  ::2] * ab_weights[0]
    img_raw_wb[1::2,  ::2] = img_raw_wb[1::2,  ::2] * ab_weights[1]
    img_raw_wb[ ::2, 1::2] = img_raw_wb[ ::2, 1::2] * ab_weights[2]
    img_raw_wb[1::2, 1::2] = img_raw_wb[1::2, 1::2] * ab_weights[3]
    img_raw_wb = saturate(img_raw_wb, 0, 2**bits)

    # Demosaicing
    img_raw_pad = np.pad(img_raw_wb, 2)
    img_raw_pad[  :2,  : ] = img_raw_pad[ 2: 4,  :  ]
    img_raw_pad[-2: ,  : ] = img_raw_pad[-4:-2,  :  ]
    img_raw_pad[  : ,  :2] = img_raw_pad[  :  , 2: 4]
    img_raw_pad[  : ,-2: ] = img_raw_pad[  :  ,-4:-2]
    img_rgb = demosaic(img_raw_pad, 'BGGR')[2:-2,2:-2] / 2**(bits - 8)
    img_rgb = saturate(img_rgb, 0, 255)

    # Color correction
    img_rgb_ccm = np.einsum('ij,...j', ccm_matrix, img_rgb, optimize=True)
    img_rgb_ccm = saturate(img_rgb_ccm, 0, 255)

    # Gamma correction
    img_rgb_gamma = 255 * (img_rgb_ccm / 255)**gamma

    return saturate(img_rgb_gamma, 0, 255).astype(np.uint8)

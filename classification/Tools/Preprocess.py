
import cv2
import numpy as np
from skimage import exposure

def histogramEquiRGB(Img, grid = 15, clip_limit=0.05):
    Img = exposure.adjust_log(Img, 1)
    Img_yuv2 = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(grid, grid))
    Img_yuv2[:, :, 0] = clahe.apply(Img_yuv2[:, :, 0] )
    ResultImg = cv2.cvtColor(Img_yuv2, cv2.COLOR_YCrCb2BGR)
    return ResultImg


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel() #
    template = template.ravel() #

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    result = np.uint8(interp_t_values[bin_idx].reshape(oldshape))
    return result


def hist_match_frontMask(source, sourceMask,  template, templateMask):
    oldshape = source.shape
    source0 = source[sourceMask>0, :]#.ravel() #
    template0 = template[templateMask>0, :]#.ravel() #
    # get the set of unique pixel values and their corresponding indices and
    # counts

    ResultImg = np.zeros(oldshape)

    for i in range(3):
        source = source0[:, i]
        template = template0[:,i]

        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        # print(s_values, bin_idx, s_counts)
        t_values, t_counts = np.unique(template, return_counts=True)
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        # print(len(interp_t_values))
        ResultImg[sourceMask>0, i] = np.uint8(interp_t_values[bin_idx])

    ResultImg = np.uint8(ResultImg)
    return ResultImg


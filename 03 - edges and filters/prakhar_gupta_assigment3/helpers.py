import cv2 as cv
import numpy as np
import warnings
from typing import Optional


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Converts a uint8 image to a float image in the range [0.0, 1.0].
    """
    if image.dtype.kind == 'f':
        return image
    elif image.dtype == np.uint8:
        return image / 255.0
    else:
        raise TypeError(f"Unsupported image type: {image.dtype}")


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Converts a float image in the range [0.0, 1.0] to a uint8 image.
    """
    if image.dtype == np.uint8:
        return image
    if image.dtype.kind == 'f':
        return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {image.dtype}")


def non_maximal_suppression(values: np.ndarray,
                            out: Optional[np.ndarray] = None,
                            window: int = 5) -> np.ndarray:
    """
    Performs non-maximal suppression on the given values matrix. The window size is
    the size of the neighbourhood to consider around each pixel.

    The value of the output matrix is equal to the value of the input "values" matrix
    everywhere that the inputs value is the local maximum within a window x window
    neighborhood. Everywhere else, the output value is zero.

    The values matrix is not modified, but a new matrix is returned, or the 'out' matrix
    is used if provided.
    """
    '''
    Code below has been developed using chatgpt and modified to work with the assigment
    problem
    
    '''
    out = np.zeros_like(values) if out is None else out
    ... # YOUR CODE HERE

    half_window = window // 2
    padded_values = np.pad(values, half_window, mode='constant',constant_values=(0))
    # print(out.shape)
    # import matplotlib.pyplot as plt
    # plt.hist(padded_values[padded_values>=200].ravel())
    # plt.show()
    r=0
    for i in range(half_window, values.shape[0] + half_window):
        c=0
        for j in range(half_window, values.shape[1] + half_window):
            neighborhood = padded_values[i - half_window:i + half_window + 1, 
                                        j - half_window:j + half_window + 1]
            
            # Check if the central value is the maximum within the neighborhood
            center=neighborhood[neighborhood.shape[0]//2,neighborhood.shape[1]//2]
            # print(center)
            if center >= neighborhood.max() and neighborhood.max()!=0:
                
                out[r,c] = 255  # Set the center element to 1 if it's the maximum
            c+=1
        r+=1    
    return out



def conv2D(f: np.ndarray,
           h: np.ndarray,
           borderType: int = cv.BORDER_REPLICATE) -> np.ndarray:
    """2D convolution (whereas cv.filter2D is correlation). Convolution is equivalent
    to correlation with a flipped filter.
    """
    f, h = uint8_to_float(f), uint8_to_float(h)
    if is_separable(h):
        u, v = split_separable_filter(h)
        f_u = cv.filter2D(f, -1, u, borderType=borderType)
        return cv.filter2D(f_u, -1, v, borderType=borderType)
    else:
        return cv.filter2D(f, -1, cv.flip(h, -1), borderType=borderType)


def split_separable_filter(h:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a separable filter, return the two 1D filters that make it up. If the given
    filter is not separable, this function returns the best approximation of it and
    displays a warning.

    Returns: (vertical_part, horizontal_part) where each is a 1D filter. The vertical
    part is a column vector and the horizontal part is a row vector.
    """
    if h.ndim != 2:
        raise ValueError("Filter must be 2D.")
    if not is_separable(h):
        warnings.warn("Filter is not separable. Using best approximation.")
    u, s, vh = np.linalg.svd(h)
    vertical_part = u[:, :1] * np.sqrt(s[:1])
    horizontal_part = vh[:1, :] * np.sqrt(s[:1])
    return vertical_part, horizontal_part


def is_separable(h:np.ndarray, tolerance: float = 1e-6) -> bool:
    """Return whether a given 2D filter is separable within a given tolerance.
    """
    if h.ndim != 2:
        raise ValueError("Filter must be 2D.")
    u, s, vh = np.linalg.svd(h)
    sum_of_singular_values = np.sum(s[1:])
    return sum_of_singular_values < tolerance


def filter_filter(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Return a new filter such that convolving with h1 then convolving with h2 is
    equivalent to just convolving with the combined filter.

    In other words, return h3 such that
        h3 = filter_filter(h1, h2).
        conv2D(conv2D(f, h1), h2) == conv2D(f, h3)

    Under the hood, this function makes use of the fact that convolution is associative
    and constructs h3 by convolving h1 with h2.
    """
    if h1.ndim != 2 or h2.ndim != 2:
        raise ValueError("Filters must be 2D (but they may be column or row vectors).")
    # Need to pad h1 with zeros on each side to "make room" for the filter h2
    m, n = h2.shape[:2]
    h1_padded = cv.copyMakeBorder(h1,
                                  top=m//2,
                                  bottom=m//2,
                                  left=n//2,
                                  right=n//2,
                                  borderType=cv.BORDER_CONSTANT,
                                  value=0)
    return conv2D(h1_padded, h2)

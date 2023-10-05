import numpy as np
import cv2 as cv
from pathlib import Path
from helpers import float_to_uint8, uint8_to_float
import argparse


def frequency_coordinates_yx(img_shape: tuple) -> (np.ndarray, np.ndarray):
    """Get the frequency coordinates for an image of a given size. In other words, the outputs are arrays f_y and f_x
    such that f_y[i, j] is the vertical frequency and f_x[i, j] is the horizontal frequency of the sinusoid denoted by
    im_f[i, j] where im_f is the fourier transform of an image of size img_shape.

    For more information, see np.fft.fftfreq.
    """
    return np.meshgrid(np.fft.fftfreq(img_shape[0]), np.fft.fftfreq(img_shape[1]))


def low_pass(img: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies a low pass filter to the image by masking out frequencies higher than cutoff in the fourier domain.

    :param img: the image to filter
    :param cutoff: the cutoff frequency (as a fraction of the width of the image)
    :return: the filtered image
    """
    ...  # YOUR CODE HERE
    imgf = np.fft.fft2(img, axes=(0, 1))

    # print(imgf.shape)

    f_y, f_x = frequency_coordinates_yx(imgf.shape)
    # print(f_y.max(), f_x.max())
    # print(f_y.min(), f_x.min())

    distance = np.sqrt((f_x) ** 2 + (f_y) ** 2)

    # Create a mask to retain low-frequency components
    mask = distance <= cutoff
    # print(mask.shape)
    mask = np.stack([mask] * img.shape[2], axis=-1)

    # Apply the mask to the Fourier-transformed image
    imgf_filtered = imgf * mask

    # Inverse Fourier transform to get the filtered image
    filtered_img = np.fft.ifft2(imgf_filtered, axes=(0, 1)).real

    return filtered_img


def high_pass(img: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies a high pass filter to the image by masking out frequencies lower than cutoff in the fourier domain.

    :param img: the image to filter
    :param cutoff: the cutoff frequency (as a fraction of the width of the image)
    :return: the filtered image
    """
    ...  # YOUR CODE HERE

    imgf = np.fft.fft2(img, axes=(0, 1))

    # print(imgf.shape)

    f_y, f_x = frequency_coordinates_yx(imgf.shape)
    # print(f_y.max(), f_x.max())
    # print(f_y.min(), f_x.min())

    distance = np.sqrt((f_x) ** 2 + (f_y) ** 2)
    # Create a mask to retain low-frequency components
    mask = distance <= cutoff
    # print(mask.shape)
    mask = np.stack([mask] * img.shape[2], axis=-1)
    mask = 1 - mask
    # Apply the mask to the Fourier-transformed image
    imgf_filtered = imgf * mask

    # Inverse Fourier transform to get the filtered image
    filtered_img = np.fft.ifft2(imgf_filtered, axes=(0, 1)).real

    return filtered_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image1", type=Path)
    parser.add_argument("image2", type=Path)
    parser.add_argument("--low-cutoff", type=float, required=True)
    parser.add_argument("--high-cutoff", type=float, required=True)
    args = parser.parse_args()

    if not args.image1.exists():
        print(f"Image {args.image1} not found")
        exit(1)

    if not args.image2.exists():
        print(f"Image {args.image2} not found")
        exit(1)

    img1 = uint8_to_float(cv.imread(str(args.image1)))
    img1_low_pass = low_pass(img1, args.low_cutoff)
    cv.imwrite(
        f"output_images/{args.image1.stem}_low_pass.jpg", float_to_uint8(img1_low_pass)
    )

    img2 = uint8_to_float(cv.imread(str(args.image2)))
    img2_high_pass = high_pass(img2, args.high_cutoff)
    cv.imwrite(
        f"output_images/{args.image2.stem}_high_pass.jpg",
        float_to_uint8(img2_high_pass),
    )

    img_hybrid = img1_low_pass + img2_high_pass
    cv.imwrite(
        f"output_images/{args.image1.stem}_{args.image2.stem}_hybrid.jpg",
        float_to_uint8(img_hybrid),
    )

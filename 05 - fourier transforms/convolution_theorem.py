import numpy as np
import cv2 as cv
from pathlib import Path
from helpers import conv2D
import argparse


# THIS FILE IS INTENTIONALLY BLANK – SEE INTSRUCTIONS IN MD FILE. IMPORTS AT THE TOP ARE JUST TO HELP YOU GET STARTED.
# FOR REFERENCE, THE "ANSWER KEY" WAS ABOUT 40 LINES OF CODE INCLUDING IMPORTS, WHITESPACE, ARGPARSE STUFF, ETC.

# def read_filter(file):


def load_filter(filter_path):
    """Load a filter from a text file into a 2D numpy array."""
    with open(filter_path, "r") as file:
        filter_data = [[float(val) for val in line.split()] for line in file]
    return np.array(filter_data)


def convolve_spatial(image, kernel):
    """
    Convolution in spatial

    """
    return conv2D(image, kernel)


def convolve_frequency(image, kernel):
    """
    Convolution in frequency

    """
    # Compute the Fourier Transform of the image
    img_f = np.fft.fft2(image, axes=(0, 1))

    # filter transform
    filter_f = np.fft.fft2(kernel, s=image.shape[:2], axes=(0, 1))
    # filter 3 channel
    filter_f = np.stack([filter_f] * img_f.shape[2], axis=-1)
    # Convoltion is just bit wise multiplication in freq domain
    result_f = img_f * filter_f
    result = np.fft.ifft2(result_f, axes=(0, 1)).real

    return result


def main(args):
    # Load the input image
    image_path = Path(args.image)
    image = cv.imread(str(image_path))

    # Load the filter from the text file
    filter_path = Path(args.filter)
    kernel = load_filter(filter_path)

    if args.mode == "spatial":
        result = convolve_spatial(image, kernel)
    else:
        result = convolve_frequency(image, kernel)
    result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX)

    # Save the filtered image
    output_path = Path("output_images") / (image_path.stem + "_new_filtered.jpg")
    cv.imwrite(str(output_path), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform convolution in spatial or frequency domain."
    )
    parser.add_argument("--image")
    parser.add_argument("--filter")
    parser.add_argument("--mode", choices=["spatial", "frequency"], default="spatial")

    args = parser.parse_args()
    main(args)

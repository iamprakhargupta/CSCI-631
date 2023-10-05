import cv2 as cv
import numpy as np
from pathlib import Path
import argparse


def fourier_magnitude_as_image(img_f: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Displays the log magnitude of the fourier transform of the image with the DC component in the center.

    :param img_f: fourier transform of an image
    :return: a BGR or grayscale image representing the (log) magnitude of the fourier transform.
    """
    ...  # YOUR CODE HERE
    img_f = np.fft.fftshift(img_f)
    mag = np.abs(img_f)

    mag = np.log(mag)
    mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    return mag


def fourier_phase_as_image(img_f: np.ndarray) -> np.ndarray:
    """Displays the phase of the fourier transform of the image with the DC component in the center, using HSV color
    space (saturation and value both set to max).

    :param img_f: fourier transform of an image
    :return: a BGR image representing the phase of the fourier transform with hue.
    """
    ...  # YOUR CODE HERE
    x, y = img_f.shape
    img_f = np.fft.fftshift(img_f)
    phase = np.angle(img_f)
    phase = np.degrees(phase)
    # print(phase.min(),phase.max())
    phase = (phase - phase.min()) / (phase.max() - phase.min())
    hue = (phase * 180).astype(np.uint8)
    saturation = np.ones_like(hue) * 255
    value = np.ones_like(hue) * 255

    # Create an HSV image with hue, saturation, and value
    hsv_image = cv.merge([hue, saturation, value])

    # Convert HSV image to BGR
    bgr_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    return bgr_image


def make_mag_phase_images(image_file: Path):
    file_mag = Path("output_images") / f"{image_file.stem}_mag.png"
    file_phase = Path("output_images") / f"{image_file.stem}_phase.png"

    img = cv.imread(str(image_file), cv.IMREAD_GRAYSCALE)

    # Do the fourier transform
    img_f = np.fft.fft2(img)

    img_mag = fourier_magnitude_as_image(img_f)
    cv.imwrite(str(file_mag), img_mag)

    img_phase = fourier_phase_as_image(img_f)
    cv.imwrite(str(file_phase), img_phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images",
        help="One or more images (space-separated if more than one) to process",
        type=Path,
        nargs="+",
    )
    args = parser.parse_args()

    for image in args.images:
        if not image.exists():
            print(f"Image {image} not found")
            continue

        make_mag_phase_images(image)

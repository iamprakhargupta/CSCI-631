import cv2 as cv
import numpy as np
import argparse
from pathlib import Path


###############################################
# Place your custom function definitions here #
###############################################
def adjust_image(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    '''
    Parts of this code is generated using ChatGPT and
    modified to align with the assigment
    
    This code
    Does gamma correction
    Does r and b scaling
    Equalizes the l channel after converting rgb to lab
    Returns a RGB image
    '''

    b, g, r = cv.split(image)
    gamma = 1/0.45 ## gamma value from CVAA book

    # Apply gamma correction to the red and blue channels
    gamma_corrected_r = np.power(r / 255.0, gamma) * 255
    gamma_corrected_b = np.power(b / 255.0, gamma) * 255

    # Scale the gamma-corrected red channel by 120% and the blue channel by 80%
    scaled_r = cv.convertScaleAbs(gamma_corrected_r, alpha=1.2)
    scaled_b = cv.convertScaleAbs(gamma_corrected_b, alpha=0.8)

    image = cv.merge([scaled_b, g, scaled_r])
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab_image)
    equalized_l = cv.equalizeHist(l)
    scaled_lab_image = cv.merge([equalized_l, a, b])
    image = cv.cvtColor(scaled_lab_image, cv.COLOR_LAB2BGR)
    return image


def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image))
    adjusted_image = adjust_image(image)
    save_name = args.image.stem + "_adjusted" + args.image.suffix
    cv.imwrite(str(save_name), adjusted_image)


# Code inside the if statement below will only be executed if the script is called
# directly from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"File {args.image} not found")
        exit(1)

    main(args)

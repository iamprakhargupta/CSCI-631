import cv2 as cv
import numpy as np
import argparse
from pathlib import Path


###############################################
# Place your custom function definitions here #
###############################################
def adjust_image(image: np.ndarray) -> np.ndarray:
    '''
    Parts of this code is generated using ChatGPT and
    modified to align with the assigment

    This Function adjusts the underwater images

    Tested with all three i think bilateral filtering 
    is the best the fine text in the book is better visible compared to all of them
    
    '''
    ... # YOUR CODE HERE
    # kernel_size = 3
    # image=cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
       
    d = 3  
    sigma_color = 75  
    sigma_space = 75  
    # kernel_size = 3
    # image = cv.medianBlur(image, kernel_size)

    image = cv.bilateralFilter(image, d, sigma_color, sigma_space)

    return image



def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image),cv.IMREAD_GRAYSCALE)
    adjusted_image = adjust_image(image)
    save_name = args.image.stem + "__denoised" + args.image.suffix
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

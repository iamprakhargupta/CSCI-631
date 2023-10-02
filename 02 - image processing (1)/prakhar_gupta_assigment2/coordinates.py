import cv2 as cv
import numpy as np
import argparse
from pathlib import Path


###############################################
# Place your custom function definitions here #
###############################################
def adjust_image(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    return image


def callback(event, x, y, flags, param):
    # event is one of cv.EVENT_* constants
    # x and y are the coordinates of the mouse click
    # flags is a bitfield of cv.EVENT_FLAG_* constants
    # param is a user-defined parameter passed to setMouseCallback

    '''
    Call back function to capture the clicked coordinates press q to exit
    '''
    if event == cv.EVENT_LBUTTONDBLCLK:
        print("Coords",x,y)
        print("Press q to exit")
        global coord1
        global coord2
        coord1,coord2=x,y

def display(image):
    """
    First will display the image 
    Then will print the coordinates where mouse clicks
    Press q to exit from both cases
    """

    cv.imshow('window title', image)
    print(image.shape)
    # Wait for the user to press the 'Q' key
    while True:
        k = cv.waitKey(1)
        if k == ord('q'):
            break
        # Close all windows
    cv.destroyAllWindows()
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    # cv.resize()
    cv.imshow('image', image)
    cv.setMouseCallback('image',callback)
    # Wait for the user to press the 'Q' key
    
    while True:
        k = cv.waitKey(1)
        if k == ord('q'):
            break
    # Close all windows
    # print(coord1,coord2)
    cv.destroyAllWindows()


def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image))
    # adjusted_image = adjust_image(image)
    # save_name = args.image.stem + "_adjusted" + args.image.suffix
    # cv.imwrite(str(save_name), adjusted_image)
    display(image=image)


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

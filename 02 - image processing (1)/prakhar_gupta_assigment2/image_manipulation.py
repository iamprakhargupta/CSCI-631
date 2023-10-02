import cv2 as cv
import numpy as np
import argparse
from pathlib import Path


###############################################
# Place your custom function definitions here #
###############################################
def callback(event, x, y, flags, param):
  # event is one of cv.EVENT_* constants
  # x and y are the coordinates of the mouse click
  # flags is a bitfield of cv.EVENT_FLAG_* constants
  # param is a user-defined parameter passed to setMouseCallback
  """
  Using coord1 and coord2 as globals var to capture mouse click
  
  """
  if event == cv.EVENT_LBUTTONDBLCLK:
    print("Coords clicked",x,y)
    print("Press q to exit")
    global coord1
    global coord2
    coord1,coord2=x,y
    




def crop_around_clicked(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    """
    This function crop around click please press q to exit
    """
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
    cv.destroyAllWindows()
    a,b=coord2-50,coord2+50
    c,d=coord1-50,coord1+50
    image=image[a:b,c:d,:]
    print(image.shape)

    return image

def scale_by_half_using_numpy_slicing(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    """
    Scales the image by half using numpy
    """
    print("full size",image.shape)
    image=image[::2,::2,:]
    print("by half size",image.shape)
    return image

def scale_by_half_using_cv_resize(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    """
    Uses cv to scale the image by half
    """
    width = image.shape[1]//2
    height = image.shape[0]//2
    dim = (width, height)
    image=cv.resize(image,dim)
    return image

def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    """
    horizontal fliping of image
    """
    image=image[:,::-1,:]
    return image

def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    '''
    Rotates counter clockwise
    '''
    print(image.shape)
    image=image.transpose(1, 0, 2)
    print(image.shape)
    return image


def swap_b_r(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    '''
    Swaps r and b channel
    '''
    b, g, r = cv.split(image)
    image=cv.merge([r, g, b])
    return image

def invert_hue_ab(image: np.ndarray) -> np.ndarray:
    ... # YOUR CODE HERE
    '''
    Does operation on the a and b channel after converting them
    from rgb
    returns an rgb image
    '''

    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab_image)
    a=255-a
    b=255-b
    image=cv.merge([l,a, b])
    image=cv.cvtColor(image, cv.COLOR_LAB2BGR)
    return image


def main(args: argparse.Namespace) -> None:
    image = cv.imread(str(args.image))

    image1 = crop_around_clicked(image)
    cv.imwrite("clicked.jpg", image1)

    image2 = scale_by_half_using_numpy_slicing(image)
    cv.imwrite("halfsize.jpg", image2)

    image3 = scale_by_half_using_cv_resize(image)
    cv.imwrite("halfsize_cv.jpg", image3)

    image4 = horizontal_mirror_image(image)
    cv.imwrite("flipped.jpg", image4)

    image5 = rotate_counterclockwise_90(image)
    cv.imwrite("rotated.jpg", image5)

    image6 = swap_b_r(image)
    cv.imwrite("swapped.jpg", image6)

    image7 = invert_hue_ab(image)
    cv.imwrite("inverted_ab.jpg", image7)


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

import cv2 as cv
import numpy as np
import argparse
from pathlib import Path
from helpers import conv2D,non_maximal_suppression
import matplotlib.pyplot as plt


def my_edge_detect(image: np.ndarray, filter_x: np.ndarray, filter_y: np.ndarray):
    # Get gradients in x and y directions
    grad_x = conv2D(image, filter_x)
    grad_y = conv2D(image, filter_y)

    # Get magnitude and direction of edges
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)

    return magnitude, direction


def edge_display(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Given magnitude and direction of edges, create a color image to display them.

    The direction of the edges is encoded as the hue of the HSV image, and the
    magnitude is encoded as the value. The saturation is always 1.

    Assumes magnitude has nonnegative values and direction has values in radians.
    """
    magnitude = magnitude / magnitude.max()
    hue = direction % (2 * np.pi) / (2 * np.pi) * 180 / 255
    saturation = np.ones_like(magnitude)
    value = 0.1 + 0.9 * magnitude
    hsv_float = np.stack([hue, saturation, value], axis=-1)
    bgr = cv.cvtColor(np.clip(hsv_float * 255, 0, 255).astype("uint8"), cv.COLOR_HSV2BGR)
    return bgr


def main(image, args,save_file=None):
    # Note that these filters are "flipped" from their usual definition because
    # we're using convolution instead of correlation.
    filter_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=np.float32)
    filter_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], dtype=np.float32)

    # YOUR CODE HERE. You need to decide on whatever other pre-processing you want to
    # do either to the image (e.g. downsampling) or to the filter (e.g. adding
    # Gaussian blur). These values may change depending on the image you're using,
    # so you may want to make them command-line arguments and parameters to this
    # function. The helpers.filter_filter function may be useful.
    ...

    '''
    Code below has been developed using chatgpt and modified to work with the assigment
    problem
    
    '''
    print(args)
    blur=args.blur
    gaussian=int(args.gaussian)
    gaussian_sigma=float(args.gaussian_sigma)
    bilateral=args.bilateral.split(",")
    post_processing=args.post_processing
    threshold=float(args.threshold)
    nms_window=int(args.nms_window)
    normalize=args.normalize
    # print(type(threshold))
    if normalize =="minmax":
        cv.normalize(image,image,0,1,cv.NORM_MINMAX,-1)
    # image=image/255.0


    if blur != "none":
        if blur == "gaussian":
            image = cv.GaussianBlur(image,(gaussian,gaussian),gaussian_sigma)
        elif blur == "bilateral":
            a,b,c=bilateral
            image=cv.bilateralFilter(image,int(a),int(b),int(c))



    # Run the edge detector to get magnitude and direction of edges in the image
    mag, dir = my_edge_detect(image, filter_x, filter_y)

    # YOUR CODE HERE. Do you want to do any post-processing on the magnitude and
    # direction? For example, if you downsampled the original image, you could
    # upsample mag and dir to match the original size. You could add a 'threshold'
    # parameter and reject any magnitude values (set them to zero) if they are below
    # the threshold. What does non-maximal suppression on the magnitude do to the
    # result? Explore some options to get the best edge images you can.



    mag[mag<threshold]=mag.min()
        
    if post_processing == "threshold_nms":
        mag=non_maximal_suppression(mag,window=nms_window)

    # print(mag.shape)
    # mag=non_maximal_suppression(mag)



    

    # Convert the magnitude and direction to a color image where magnitude is brightness
    # and direction is hue.
    edge_image = edge_display(mag, dir)

    # If a save file is given, write the output there. Otherwise, display it on screen.
    if save_file is not None:
        cv.imwrite(str(save_file), edge_image)
    else:
        cv.imshow("Edges", edge_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--algorithm", default="sobel", choices=["sobel", "dog"])
    parser.add_argument("--normalize", default="minmax", choices=["none", "minmax"])
    parser.add_argument("--blur", default="none", choices=["bilateral", "gaussian","none"])
    parser.add_argument("--gaussian", default=5)
    parser.add_argument("--gaussian_sigma", default=5)
    parser.add_argument("--bilateral", default="9,75,75")
    parser.add_argument("--post_processing", default="threshold", choices=["threshold", "threshold_nms"])
    parser.add_argument("--threshold", default=0.2)
    parser.add_argument("--nms_window", default=5)
    parser.add_argument("--output_file_name", default=None)
    ... # YOU MAY ADD MORE COMMAND-LINE ARGUMENTS HERE
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image file {args.image} does not exist.")
        exit(1)

    if args.output_file_name is None:
        save_file = args.image.parent / f"{args.image.stem}-out.jpg"
    else:
        save_file=args.output_file_name    
    image = cv.imread(str(args.image), cv.IMREAD_GRAYSCALE)

    # ANY COMMAND LINE ARGUMENTS YOU ADD SHOULD BE PASSED TO THE MAIN() FUNCTION
    main(image,args, save_file=save_file)

import cv2 as cv
import numpy as np
from helpers import non_maximal_suppression
from pathlib import Path
from typing import Optional

def euclidean_distance(matrix1, matrix2):
    # Calculate the element-wise squared differences
    squared_diff = (matrix1 - matrix2) ** 2
    
    # Sum along the appropriate axis (axis=1 for m x 2 matrices)
    sum_squared_diff = np.sum(squared_diff, axis=1)
    
    # Take the square root to get the Euclidean distance
    distance = np.sqrt(sum_squared_diff)
    
    return distance

def distance_to_circles(centers_xy, xy, r):
    distance=euclidean_distance(centers_xy,xy)
    return np.abs(r-distance)




class HoughCircleDetector:
    def __init__(self,
                 image_shape: tuple[int, int],
                 resolution: int,
                 radius: float,
                 soft_vote_sigma: Optional[float] = None):
        h, w = image_shape

        # Create a grid of parameters (x and y centers of circles)
        self.center_x = np.linspace(0, w-1, resolution + 1)
        self.center_x = (self.center_x[:-1] + self.center_x[1:]) / 2
        self.center_y = np.linspace(0, h-1, resolution + 1)
        self.center_y = (self.center_y[:-1] + self.center_y[1:]) / 2

        self.radius = radius
        if soft_vote_sigma is None:
            # Sensible default for 'sigma' is the diagonal length of the image divided by the resolution
            self.sigma = np.sqrt(w**2 + h**2) / resolution
        else:
            self.sigma = soft_vote_sigma

        # Initialize self.accumulator to be a 2D array of zeros with the same shape as the parameter space. The value at
        # accumulator[i,j] represents the total number of "votes" for a circle centered at (center_y[i], center_x[j])
        self.accumulator = np.zeros(shape=(len(self.center_y), len(self.center_x)), dtype=float)
        print(self.accumulator.shape)

    def clear(self):
        self.accumulator = np.zeros(shape=(len(self.center_y), len(self.center_x)), dtype=float)

    def add_edge_at_xy(self, xy: tuple[float, float]):
        """Add an edge at the given (x, y) coordinate in image-space. Add a value to the accumulator for all circles
         that pass through this point, using a soft fall-off such that points that are "close" to the circle get a
         fraction of a vote.

         More precisely, the number of votes given to a circle with center at (cx, cy) is equal to

            np.exp(-dr**2 / self.sigma**2)

         where dr is the smallest distance between (x, y) and the circle centered at (cx, cy).
        """     
        
        
        xy=xy[::-1]
        for index,i in enumerate(self.center_y):
            centers_xy = np.column_stack((np.full(self.center_x.shape, i),self.center_x))
            dr=distance_to_circles(centers_xy, xy, self.radius)
            votes=np.exp(-dr**2 / self.sigma**2)
            self.accumulator[index,:]=self.accumulator[index,:]+votes


    def get_circles(self, threshold: float, nms_window: int) -> np.ndarray:
        """Return a list of circles (cx, cy) which have a vote count above the threshold and are a local maximum in
        the accumulator space.

        :param threshold: minumum number of 'votes', as a fraction of the maximum number of votes
        :param nms_window: window size for non-maximal suppression
        :return: numpy array of shape (num_circles, 2) where each row is (x, y) coordinate of the center of the circle
        """
        # YOUR CODE HERE. YOU MUST RETURN A NUMPY ARRAY OF SHAPE (num_circles, 2) WHERE EACH ROW IS (x, y) COORDINATE
        # OF THE CENTER OF THE CIRCLE.
        
        votes = self.accumulator / self.accumulator.max()
        votes[votes < threshold] = 0
        votes = non_maximal_suppression(votes, window=nms_window)
        ii, jj = np.where(votes > 0)

        # selected_lines = np.abs(self.angles[jj] - np.pi / 2) < self.hori  # Adjust the tolerance as needed

        # # Get the offsets and angles of selected lines
        # selected_offsets = self.offsets[ii][selected_lines]
        # selected_angles = self.angles[jj][selected_lines]

        return np.stack([self.center_x[jj], self.center_y[ii]], axis=-1)


def main(image: np.ndarray,
         canny_blur: int,
         canny_threshold_1: float,
         canny_threshold_2: float,
         accumulator_threshold: float,
         nms_window: int,
         resolution: int,
         radius: float) -> np.ndarray:
    annotated_image = image.copy()

    # Convert to grayscale.
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(image.shape)
    # Apply Gaussian blur to the image before running Canny edge detection.
    image = cv.GaussianBlur(image, (canny_blur, canny_blur), 0)

    # Run Canny edge detection.
    edges = cv.Canny(image, canny_threshold_1, canny_threshold_2)
    # cv.imshow('Edges', edges)
    
    # # Wait for a key press and then close the window
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # Create a HoughCircleDetector object.


    hough = HoughCircleDetector((image.shape[0], image.shape[1]), resolution, radius)

    # Iterate over the edges and add each edge to the HoughCircleDetector.
    for y, x in np.argwhere(edges > 0):
        hough.add_edge_at_xy((x, y))
    im=hough.accumulator
    # cv.namedWindow('Image',cv.WINDOW_NORMAL)
    # cv.imshow('Image', im)
    
    # # Wait for a key press and then close the window
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Get the circles from the HoughCircleDetector.
    circles = hough.get_circles(accumulator_threshold, nms_window)
    print(circles.shape)
    # print(circles)
    # Draw the circles on the original image.
    for cx, cy in circles:
        cv.circle(annotated_image, (int(cx), int(cy)), int(radius + 0.5), (0, 0, 255), 1, cv.LINE_AA)

    return annotated_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image",
                        type=Path,
                        help="Path to image file")
    parser.add_argument("--accumulator-threshold",
                        type=float,
                        default=0.9,
                        help="Threshold for circle detection, relative to max # of votes")
    parser.add_argument("--nms-window",
                        type=int,
                        default=5,
                        help="Window size for non-maximal suppression")
    parser.add_argument("--resolution",
                        type=int,
                        default=100,
                        help="Number of parameters for each of center x and center y")
    parser.add_argument("--radius",
                        type=float,
                        default=30,
                        help="Radius of circles to detect")
    parser.add_argument("--canny-blur",
                        type=float,
                        default=3,
                        help="Amount of Gaussian blur to apply before Canny edge detection")
    parser.add_argument("--canny-threshold-1",
                        type=float,
                        default=100,
                        help="Low threshold for Canny edge detection")
    parser.add_argument("--canny-threshold-2",
                        type=float,
                        default=200,
                        help="High threshold for Canny edge detection")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Could not find image file: {args.image}")

    out_file = args.image.parent / "output_images" / f"{args.image.stem}_{int(args.radius)}.jpg"

    kwargs = vars(args)
    kwargs["image"] = cv.imread(str(args.image))
    image_with_circles = main(**kwargs)

    cv.imwrite(str(out_file), image_with_circles)

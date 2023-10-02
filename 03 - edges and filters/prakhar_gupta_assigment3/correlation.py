import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional


def my_correlation(image: np.ndarray,
                   kernel: np.ndarray,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Performs correlation of the given image with the given kernel, without padding.
    """
    '''
    Code below has been developed using chatgpt and modified to work with the assigment
    problem
    
    '''
    out = np.zeros_like(image) if out is None else out
    ... # YOUR CODE HERE
    # Get the dimensions of the image and filter
    image_height, image_width, _ = image.shape ## 3 channel image
    filter_height, filter_width = kernel.shape



    # Iterate over each pixel in the original image
    for i in range(image_height):
        for j in range(image_width):
            convolution = 0
            
            # Iterate over the filter
            for m in range(filter_height):
                for n in range(filter_width):
                    # Calculate the corresponding image coordinates
                    x = i + m - filter_height // 2
                    y = j + n - filter_width // 2
                    
                    # Apply BORDER_REPLICATE behavior to handle border pixels
                    x = max(0, min(x, image_height - 1))
                    y = max(0, min(y, image_width - 1))
                    
                    convolution += image[x, y,:] * kernel[m, n]
            
            out[i, j] = convolution

    return out


def run_and_time_filters(image, kernel_size: int = 5):
    gaussian_kernel_1d = cv.getGaussianKernel(kernel_size, -1)
    gaussian_kernel_1d = gaussian_kernel_1d / np.sum(gaussian_kernel_1d)
    gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

    start = time.time()
    cv_result_2d = cv.filter2D(image, -1, gaussian_kernel_2d, borderType=cv.BORDER_REPLICATE)
    elapsed_cv_2d = time.time() - start
    print(f"2D filter time (OpenCV, {kernel_size}x{kernel_size}): {elapsed_cv_2d}")

    start = time.time()
    cv_result_vertical = cv.filter2D(image, -1, gaussian_kernel_1d, borderType=cv.BORDER_REPLICATE)
    cv_result_separable = cv.filter2D(cv_result_vertical, -1, gaussian_kernel_1d.T, borderType=cv.BORDER_REPLICATE)
    elapsed_cv_separable = time.time() - start
    print(f"Separable filter time (OpenCV, {kernel_size}x{kernel_size}): {elapsed_cv_separable}")

    start = time.time()
    my_result_2d = my_correlation(image, gaussian_kernel_2d)
    elapsed_mine_2d = time.time() - start
    print(f"2D filter time (Mine, {kernel_size}x{kernel_size}): {elapsed_mine_2d}")

    start = time.time()
    my_result_vertical = my_correlation(image, gaussian_kernel_1d)
    my_result_separable = my_correlation(my_result_vertical, gaussian_kernel_1d.T)
    elapsed_mine_separable = time.time() - start
    print(f"Separable filter time (Mine, {kernel_size}x{kernel_size}): {elapsed_mine_separable}")

    # Inspect results and/or assert that the results are the same within sensible tolerances
    ... # YOUR CODE HERE
    # print(cv_result_2d.shape)
    print("Difference between 2D filter is more than 0.1 per pixel?")
    diff=np.abs((my_result_2d/255.0)-(cv_result_2d/255.0))
    flag=np.greater(diff, 0.1)
    if flag.sum()==0:
        print("Nope its not")\
        
    print("Difference between Separable filter is more than 0.1 per pixel?")
    diff=np.abs((my_result_separable/255.0)-(cv_result_separable/255.0))
    flag=np.greater(diff, 0.1)
    if flag.sum()==0:
        print("Nope its not")    
    


    return elapsed_cv_2d, elapsed_cv_separable, elapsed_mine_2d, elapsed_mine_separable


def main():
    # Re-use one of the images from Problem 3 here, just for convenience.
    image = cv.imread("edges03.jpg")

    # Crop it down to a more manageable size for this experiment
    image = image[:200, :200, :]

    ... # YOUR CODE HERE (run for a variety of filter sizes and plot the results)
    import matplotlib.pyplot as plt
    filterbank=[3+i for i in range(0,10,2)]
    elapsed_cv_2d_list=[]
    elapsed_cv_separable_list=[]
    elapsed_mine_2d_list=[]
    elapsed_mine_separable_list=[]

    # print(filterbank)
    for i in filterbank:
        print("############Filter Size#############",i)
        elapsed_cv_2d, elapsed_cv_separable, elapsed_mine_2d, elapsed_mine_separable=run_and_time_filters(image=image,kernel_size=i)
        elapsed_cv_2d_list.append(elapsed_cv_2d)
        elapsed_cv_separable_list.append(elapsed_cv_separable)
        elapsed_mine_2d_list.append(elapsed_mine_2d)
        elapsed_mine_separable_list.append(elapsed_mine_separable)

    fig=plt.figure(figsize=(6, 4))
    plt.plot(filterbank, elapsed_cv_2d_list, marker='o', label='cv_2d')  
    plt.plot(filterbank, elapsed_cv_separable_list, marker='o', label='cv_separable')
    plt.plot(filterbank, elapsed_mine_2d_list, marker='o', label='mine_2d')
    plt.plot(filterbank, elapsed_mine_separable_list, marker='o', label='mine_separable')
    plt.xscale('log')
    plt.yscale('log')    

    plt.xlabel('Filter Size')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    fig.savefig("correlation_runtime.png")

if __name__ == "__main__":
    main()

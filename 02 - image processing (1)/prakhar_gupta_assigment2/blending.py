import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

def blending(image1,image2,k=3):
    """
    Parts of this code is generated using ChatGPT and
    modified to align with the assigment


    This code blends image 1 and image 2 
    k is for levels

    """  
    # Generate Gaussian pyramids for the input images and the mask
    gaussian_pyr1 = [image1]
    gaussian_pyr2 = [image2]
    height, width, _=image1.shape
    print("Making 3 channel Mask")
    mask = np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the blending region width (adjust as needed)
    blend_width = 25

    # Create the linear mask
    for x in range(width):
        if x < blend_width:
            mask[:, x] = 0  # Left image dominance
        elif x > width - blend_width:
            mask[:, x] = 255  # Right image dominance
        else:
            # Linear gradient transition
            mask[:, x] = int((255 * (x - blend_width) / (width - 2 * blend_width)))



    # mask[:, :width // 2] = 255
    
    # 3 Channel Mask
    t_mask = np.zeros((height, width, 3), dtype=np.uint8)
    t_mask[:, :, 0] = mask 
    t_mask[:, :, 1] = mask 
    t_mask[:, :, 2] = mask
    mask=t_mask
    # print(mask.shape)
    gaussian_mask = [mask]
    print("Making gaussian Pyramid")

    for i in range(k):  # You can adjust the number of levels as needed
        image1 = cv.pyrDown(image1)
        image2 = cv.pyrDown(image2)
        gaussian_pyr1.append(image1)
        gaussian_pyr2.append(image2)
    
    # Generate Laplacian pyramids for the Gaussian pyramids
    laplacian_pyr1 = [gaussian_pyr1[-1]]
    laplacian_pyr2 = [gaussian_pyr2[-1]]    

    for i in range(k):
        mask = cv.pyrDown(mask)
        gaussian_mask.append(mask.astype(np.float32) / 255.0)


    print("Making laplacian Pyramid")
    for i in range(k-1, 0, -1):
        expand1 = cv.pyrUp(gaussian_pyr1[i])
        expand2 = cv.pyrUp(gaussian_pyr2[i])
        gaussian_pyr1[i-1]=cv.resize(gaussian_pyr1[i-1],(expand1.shape[1],expand1.shape[0]))
        laplacian1 = cv.subtract(gaussian_pyr1[i-1], expand1)
        gaussian_pyr2[i-1]=cv.resize(gaussian_pyr2[i-1],(expand2.shape[1],expand2.shape[0]))
        laplacian2 = cv.subtract(gaussian_pyr2[i-1], expand2)

        laplacian_pyr1.append(laplacian1)
        laplacian_pyr2.append(laplacian2)

    # Combine the Laplacian pyramids with the bitmask using the equation
    print("Blending using mask")
    blended_pyr = []
    for lap1, lap2, m1 in zip(laplacian_pyr1, laplacian_pyr2, gaussian_mask[::-1]):

        m1=cv.resize(m1,(lap2.shape[1],lap2.shape[0]))
        blended = (lap1 * m1) + (lap2 * (1 - m1))
        blended_pyr.append(blended)

    # Reconstruct the blended image from the Laplacian pyramid
    print("Inverse Gaussian of blended image")
    output_image = blended_pyr[0]
    for i in range(1, len(blended_pyr)):
        output_image = cv.pyrUp(output_image)
        output_image=cv.resize(output_image,(blended_pyr[i].shape[1],blended_pyr[i].shape[0]))
        output_image += blended_pyr[i]
    print("Done")    
    return output_image


def main(args: argparse.Namespace) -> None:
    image1 = cv.imread(str(args.image1))
    image2 = cv.imread(str(args.image2))
    # adjusted_image = adjust_image(image)
    image2=cv.resize(image2,(image1.shape[1],image1.shape[0]))
    # print(image1.shape,image2.shape)
    adjusted_image=blending(image1,image2)
    save_name = args.image1.stem +"_" +args.image2.stem +"_blended_step"+ args.image1.suffix
    cv.imwrite(str(save_name), adjusted_image)


# Code inside the if statement below will only be executed if the script is called
# directly from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image1", type=Path)
    parser.add_argument("image2", type=Path)
    args = parser.parse_args()

    # if not args.image.exists():
    #     print(f"File {args.image} not found")
    #     exit(1)

    main(args)    
import cv2 as cv
import numpy as np
from helpers import non_maximal_suppression
import matplotlib.pyplot as plt
import cProfile,pstats

def main():
    scene = cv.imread("mario.jpg")
    # Read the PNG including the alpha channel in the fourth plane, then split into BGR
    # and alpha parts
    template = cv.imread("coin.png", cv.IMREAD_UNCHANGED)
    template, mask = template[:, :, :3], template[:, :, 3:4]
    '''
    Code below has been developed using chatgpt and modified to work with the assigment
    problem
    
    '''
    
    res = cv.matchTemplate(scene,template,cv.TM_SQDIFF)
    res=(1-((res-np.min(res))/(np.max(res)-np.min(res)))*255).astype('uint8')
    

    # cv.imshow("image",res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print("Plotting the histogram")
    # plt.hist(res.ravel())
    # plt.show()
    print("Setting the histogram")

    threshold = 200
    res[res<threshold]=0
    print("Displaying image with coins")

    # cv.imshow("image",res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # plt.hist(res[res>=200].ravel())
    # plt.show()

    non_maximal=non_maximal_suppression(res)



    nonzero = cv.findNonZero(non_maximal)
    font = cv.FONT_HERSHEY_SIMPLEX

    fontScale = 1
    color = (0, 0, 255)
    
    thickness = 3
    scene=cv.putText(scene, str(len(nonzero)), (10,40), font, fontScale, color, thickness)
    for k in nonzero:
        i,j=k[0]
        w,h,_=template.shape
        start_point = (i-1, j-1)

        color = (0, 0, 255)
        thickness = 2
        end_point = (i+h, j+w)
        image = cv.rectangle(scene, start_point, end_point, color, thickness)
     

    # cv.imshow("image",image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    cv.imwrite("mario-matches.jpg",image)


    
    # YOUR CODE HERE (follow steps in the assignment)

if __name__ == "__main__":
    # with cProfile.Profile() as profile:
    #     main()

    # result=pstats.Stats(profile)
    # result.sort_stats(pstats.SortKey.TIME)
    # result.print_stats()
    main()    

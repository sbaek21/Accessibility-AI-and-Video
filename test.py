import cv2
import numpy as np
import os
import re

def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find first number in the filename
    return int(match.group()) if match else float('inf')  # Convert to int for sorting


def bgremove1(myimage):
    myimage =cv2.imread(myimage)
    # Blur to image to reduce noise
    myimage = cv2.GaussianBlur(myimage,(5,5), 0)
 
    # We bin the pixels. Result will be a value 1..5
    bins=np.array([0,51,102,153,204,255])
    myimage[:,:,:] = np.digitize(myimage[:,:,:],bins,right=True)*51
 
    # Create single channel greyscale for thresholding
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    # Perform Otsu thresholding and extract the background.
    # We use Binary Threshold as we want to create an all white background
    ret,background = cv2.threshold(myimage_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    # Convert mask to 3-channel image and set background to neon green
    background = np.zeros_like(myimage)
    background[:, :] = (0, 230, 0)  # Neon green background
    background[ret == 0] = (0, 230, 0)  # Ensure background is set
    # # Convert black and white back into 3 channel greyscale
    # background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("method1_background_image.png",background)

    # Perform Otsu thresholding and extract the foreground.
    # We use TOZERO_INV as we want to keep some details of the foregorund
    ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  #Currently foreground is only a mask
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    cv2.imwrite("method1_foreground_image.png",foreground)

    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground



    cv2.imwrite("output0/method1_" + str(myimage) + ".png",finalimage)
 
    return finalimage



# file_name = 'out_directory1/scene_540.jpg'
# directory_name = os.fsencode('out_directory0')
directory = 'out_directory0'
files = sorted(os.listdir(directory), key=extract_number)
for file in files:
    file_path = os.path.join(directory, file)

    if os.path.isfile(file_path):  # Ensure it's a file
        print(file_path)
        new_img = bgremove1(file_path)

# loop through all images in the directory
    

# src = cv2.imread(file_name)

# # cv2.imshow("image", new_img)
# cv2.waitKey()


## some results
# 02:12 2 mins 12 seconds 
# for 614 images
# 9 min 51 second video



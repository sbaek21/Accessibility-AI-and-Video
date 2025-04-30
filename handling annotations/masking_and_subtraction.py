import cv2
import numpy as np
import os
import re
from time import localtime, strftime
import csv

def extract_frames(video_path, sampling_frequency):
    if not os.path.exists(video_path):
        print(f"{video_path}: File not found.")
        exit()

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()

    # Extracting frames, chronologically forwards
    frames = []
    count = 0
    frame_index = 0
    while frame_index <= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frames.append(frame)

        output_path = os.path.join(mask_dir, "frame_" + str(count) + ".jpg")
        cv2.imwrite(output_path, frame)
        
        # Increment frame_index
        count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((count * fps) / sampling_frequency))

    cap.release()

    return frames, fps

def bgremove1(myimage, index):
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
    background[:, :] = (0, 0, 0)  # Black background
    background[ret == 0] = (0, 0, 0)  # Ensure background is set

    # Perform Otsu thresholding and extract the foreground.
    # We use TOZERO_INV as we want to keep some details of the foregorund
    ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  # Currently foreground is only a mask
    myimage[myimage == 0] = 1
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
    output_path = os.path.join(mask_dir, "masked_" + str(index) + ".jpg")
    cv2.imwrite(output_path, finalimage)
    return finalimage

def subtract_two_masks(curr_frame, next_frame, null_color):
    diff_squared = np.sum((curr_frame - next_frame)**2, axis=2) 
    curr_frame[diff_squared < 100] = null_color

    output_path = os.path.join(mask_dir, "subtracted_" + str(index) + ".jpg")
    cv2.imwrite(output_path, curr_frame)

def print_time():
    return strftime("%H:%M:%S", localtime())

video_path = 'data/videos/test3_cs361_2_2_2021.mp4'
mask_dir = 'output/image/CS_361'
output_csv = 'output/txt/annotations.csv'

sampling_frequency = 1

print(video_path)

print(f"Extracting frames {print_time()}")
frames_array, fps = extract_frames(video_path, sampling_frequency)

print(f"Creating masks {print_time()}")
for index in range(len(frames_array)):
    frames_array[index] = bgremove1(frames_array[index], index)

print(f"Starting subtraction {print_time()}")
for index in range(len(frames_array) - 1): 
    null_color = [0, 0, 0]
    curr_frame = frames_array[index]
    next_frame = frames_array[index + 1]
    subtract_two_masks(curr_frame, next_frame, null_color)

print(f"Counting remaining {print_time()}")
remaining_pixels = []
for index in range(len(frames_array) - 1):
    single_channel = cv2.cvtColor(frames_array[index], cv2.COLOR_BGR2GRAY)
    remaining_pixels.append(cv2.countNonZero(single_channel))

    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['index', 'frame', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'index' : index, 
                         'frame' : int((index * fps) / sampling_frequency),         # fps is not being calculated properly, not sure why
                         'value' : remaining_pixels[index]})       

print(f"Selecting frames {print_time()}")
# TODO: improve
# normalize remaining pixels
normed_rem_pixels = (remaining_pixels-np.min(remaining_pixels))/(np.max(remaining_pixels)-np.min(remaining_pixels))

# store indices 
different_image_indexes = []
index = 0
for percentile in normed_rem_pixels:
    if percentile > 0.1:        # arbitary, hardcoded value
        different_image_indexes.append(index + 1)
    index = index + 1

print(len(different_image_indexes), "frames selected: \n", different_image_indexes)

print(f"Finished {print_time()}")



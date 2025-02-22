import cv2
import numpy as np
import os
import re
from PIL import Image, ImageChops, ImageStat, ImageFilter
import skimage as ski
from skimage.metrics import structural_similarity as ssim

def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find first number in the filename
    return int(match.group()) if match else float('inf')  # Convert to int for sorting

# subtracted image
def subtract(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    diff = cv2.subtract(img1, img2)
    # cv2.imwrite("output0/method1_" + str(myimage) + ".png",finalimage)
    
    output_path = os.path.join("subtraction_output", "method1_" + os.path.basename(file_path))
    cv2.imwrite(output_path, diff)
    return diff

def compute_difference(img1, img2, index):
    img1 = Image.open(img1).convert('RGB')
    img2 = Image.open(img2).convert('RGB')
    img1 = img1.resize((1000, 1000))
    img2 = img2.resize((1000, 1000))

    # Blur to reduce noise
    img1 = img1.filter(ImageFilter.GaussianBlur(radius=1))
    img2 = img2.filter(ImageFilter.GaussianBlur(radius=1))


    # Mean difference method
    # diff = ImageChops.difference(img1, img2)
    # stat = ImageStat.Stat(diff)
    # mean_diff = sum(stat.mean) / len(stat.mean)

    # Structural similarity index method
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    ssim_value = ssim(img1_array, img2_array, channel_axis=2)

    threshold = 0.75
    print(ssim_value)
    # False for there is no difference
    if ssim_value > threshold:
        return False
    else:
        print("Difference found at: " + str(index))
        print("SSIM VALUE: " + str(ssim_value))

    return True





directory = 'output'
files = sorted(os.listdir(directory), key=extract_number)
different_image_indexes = []
count = 0
for file in files:
    # Don't want it to include last file
    if count == len(files) - 2:
        break

    count += 1

    file_path = os.path.join(directory, file)
    next_file = os.path.join(directory, files[count + 1])
    if os.path.isfile(file_path) and os.path.isfile(next_file):  # Ensure it's a file
        print(count)
        if compute_difference(file_path, next_file, count):
            different_image_indexes.append(count + 1)


print(different_image_indexes)
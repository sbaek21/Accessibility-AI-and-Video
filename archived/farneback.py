
# averge vertical motion magnitudes between frames ------------------------

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


def is_scene_change_ssim(frame1, frame2, threshold=0.6):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score < threshold

def is_vertical_scroll(vert_flow, magnitude, mag_thresh=(1.0, 5.0), direction_thresh=0.5):
    avg_vert = np.mean(vert_flow)
    direction_consistent = abs(avg_vert) > direction_thresh
    return mag_thresh[0] <= magnitude <= mag_thresh[1] and direction_consistent

# Folder containing the image sequence
image_folder = 'out_directory_crop_comb'

# Get sorted list of image filenames
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
])

# Make sure we have at least two frames
if len(image_files) < 2:
    print("Not enough images in folder to compute optical flow.")
    exit()

# Read the first image
frame1 = cv2.imread(image_files[0])
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

frame_counter = 1
step = 16

# Optional: Store difference magnitudes
vertical_motion_magnitudes = []
scroll_frame_pairs = []

for i in range(1, len(image_files)):
    # Read the next image
    frame2 = cv2.imread(image_files[i])

    # SCENE CHANGE DETECTION
    if is_scene_change_ssim(frame1, frame2, threshold=0.75):
        print(f"Frame {i-1} to {i}: Scene change detected. Skipping")
        prvs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame1 = frame2.copy()
        continue
    
    # OPTICAL FLOW
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract only the vertical component
    vert_flow = flow[..., 1]

    # --- Calculate magnitude of vertical motion ---
    vert_magnitude = np.mean(np.abs(vert_flow))
    vertical_motion_magnitudes.append(vert_magnitude)
    print(f"Frame {i-1} to {i}: Mean vertical motion = {vert_magnitude:.4f}")

    

    if is_vertical_scroll(vert_flow, vert_magnitude):
        print(f"Frame {i-1} to {i}: Vertical scrolling detected.")
        scroll_frame_pairs.append((i - 1, i))  # New list to store scroll indices

    # Draw vertical flow on the frame
    vis = frame2.copy()
    h, w = next.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            fy = vert_flow[y, x]
            end_point = (x, int(y + fy))
            cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

    # Display vertical component image
    cv2.imshow('Vertical Flow Visualization', vis)

    # # Optionally show the grayscale normalized version too
    # vert_norm = cv2.normalize(vert_flow, None, 0, 255, cv2.NORM_MINMAX)
    # vert_img = vert_norm.astype('uint8')
    # cv2.imshow('Vertical Component (Grayscale)', vert_img)

    # Wait for key
    k = cv2.waitKey(500) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite(f'verticalflow_frame{i}.png', vis)
        cv2.imwrite(f'vertical_component_frame{i}.pgm', vert_img)

    prvs = next.copy()
    frame_counter += 1

cv2.destroyAllWindows()

# # Optionally print all stored magnitudes at the end
# print("\nAll vertical motion magnitudes between frames:")
# for idx, mag in enumerate(vertical_motion_magnitudes, start=1):
#     print(f"Frame {idx-1} to {idx}: {mag:.4f}")

# Final print
print("\nAll vertical motion magnitudes (non-scene-change only):")
for idx, mag in enumerate(vertical_motion_magnitudes):
    print(f"Segment {idx}: {mag:.4f}")

print("\nScrolling frame pairs:")
for start, end in scroll_frame_pairs:
    print(f"Frame {start} → Frame {end}: Vertical scrolling")



###############LABELING

def label_motion(magnitudes, scroll_thresh=(1.3, 5.0), scene_thresh=5.0):
    labels = []

    for i in range(len(magnitudes)):
        prev = magnitudes[i - 1] if i > 0 else 0
        curr = magnitudes[i]
        next = magnitudes[i + 1] if i < len(magnitudes) - 1 else 0

        # Check for Scene Change: isolated high motion
        if curr > scene_thresh and prev < 0.5 and next < 0.5:
            labels.append("Scene Change")

        # Check for Scrolling: part of a sequence of mid/high motion
        elif (
            scroll_thresh[0] <= curr <= scroll_thresh[1]
            or (
                curr > scroll_thresh[1] and  # >5 but contextually part of scrolling
                (prev >= scroll_thresh[0] or next >= scroll_thresh[0])
            )
            or (
                curr < scroll_thresh[0] and  # low but sandwiched between scrolling
                prev >= scroll_thresh[0] and next >= scroll_thresh[0]
            )
        ):
            labels.append("Scrolling")

        else:
            labels.append("")  # No significant change

    return labels


# labels = label_motion(magnitudes)

for i, (mag, label) in enumerate(zip(magnitudes, labels)):
    print(f"Frame {i} to {i+1}: {mag:.4f} -> {label}")

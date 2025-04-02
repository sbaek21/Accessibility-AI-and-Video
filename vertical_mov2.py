# import cv2
# import numpy as np

# # Replace with your video file path
# video_path = 'croppedvideo.mp4'

# # Open video
# cap = cv2.VideoCapture(video_path)

# # Read the first frame
# ret, frame1 = cap.read()
# if not ret:
#     print("Failed to read video.")
#     cap.release()
#     exit()

# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# frame_counter = 1

# # Step size for drawing fewer flow lines for better readability
# step = 16

# while True:
#     # Read the next frame
#     ret, frame2 = cap.read()
#     if not ret:
#         break

#     next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     # Calculate optical flow
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Create a copy of the current frame for visualization
#     vis = frame2.copy()

#     # Draw flow vectors
#     h, w = next.shape
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fx, fy = flow[y, x]
#             end_point = (int(x + fx), int(y + fy))
#             cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

#     # Display result
#     cv2.imshow('Optical Flow Visualization', vis)

#     # Prepare for next iteration
#     prvs = next.copy()

#     # Handle key input
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:  # Esc to exit
#         break
#     elif k == ord('s'):
#         cv2.imwrite(f'opticalflow_frame{frame_counter}.png', vis)

#     frame_counter += 1

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Replace with your video file path
# video_path = 'croppedvideo.mp4'

# # Open video
# cap = cv2.VideoCapture(video_path)

# # Get the frames per second (fps) of the video
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_interval = int(fps)  # Process 1 frame per second

# # Read the first frame
# ret, frame1 = cap.read()
# if not ret:
#     print("Failed to read video.")
#     cap.release()
#     exit()

# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# frame_counter = 1
# step = 16

# while True:
#     # Skip frames to reach the next 1-second mark
#     for _ in range(frame_interval - 1):
#         ret = cap.grab()
#         if not ret:
#             break

#     # Read the next frame
#     ret, frame2 = cap.read()
#     if not ret:
#         break

#     next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     # Compute optical flow
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Draw flow vectors
#     vis = frame2.copy()
#     h, w = next.shape
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fx, fy = flow[y, x]
#             end_point = (int(x + fx), int(y + fy))
#             cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

#     # Display the result
#     cv2.imshow('Optical Flow Visualization', vis)

#     prvs = next.copy()

#     k = cv2.waitKey(0) & 0xff  # Wait indefinitely between each second-frame
#     if k == 27:  # ESC
#         break
#     elif k == ord('s'):
#         cv2.imwrite(f'opticalflow_frame{frame_counter}.png', vis)

#     frame_counter += 1

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import os

# # Folder containing the image sequence
# image_folder = 'out_directory_constantScrolling'

# # Get sorted list of image filenames (assuming .png or .jpg)
# image_files = sorted([
#     os.path.join(image_folder, f)
#     for f in os.listdir(image_folder)
#     if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
# ])

# # Make sure we have at least two frames
# if len(image_files) < 2:
#     print("Not enough images in folder to compute optical flow.")
#     exit()

# # Read the first image
# frame1 = cv2.imread(image_files[0])
# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# frame_counter = 1
# step = 16

# for i in range(1, len(image_files)):
#     # Read the next image
#     frame2 = cv2.imread(image_files[i])
#     next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     # Compute optical flow
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Draw flow on the frame
#     vis = frame2.copy()
#     h, w = next.shape
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fx, fy = flow[y, x]
#             end_point = (int(x + fx), int(y + fy))
#             cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

#     # Show result
#     cv2.imshow('Optical Flow Visualization', vis)
    

#     # Save with 's' or move on with delay
#     k = cv2.waitKey(500) & 0xff  # wait 500ms (half a second)
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite(f'opticalflow_frame{i}.png', vis)

#     prvs = next.copy()
#     frame_counter += 1

# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import os

# # Folder containing the image sequence
# image_folder = 'out_directory_constantScrolling'

# # Get sorted list of image filenames (assuming .png or .jpg)
# image_files = sorted([
#     os.path.join(image_folder, f)
#     for f in os.listdir(image_folder)
#     if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
# ])

# # Make sure we have at least two frames
# if len(image_files) < 2:
#     print("Not enough images in folder to compute optical flow.")
#     exit()

# # Read the first image
# frame1 = cv2.imread(image_files[0])
# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# frame_counter = 1
# step = 16

# for i in range(1, len(image_files)):
#     # Read the next image
#     frame2 = cv2.imread(image_files[i])
#     next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     # Compute optical flow
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Draw flow on the frame
#     vis = frame2.copy()
#     # h, w = next.shape
#     # for y in range(0, h, step):
#     #     for x in range(0, w, step):
#     #         fx, fy = flow[y, x]
#     #         end_point = (int(x + fx), int(y + fy))
#     #         cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

#     # # Show result
#     # cv2.imshow('Optical Flow Visualization', vis)

#     # horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
#     vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
#     # horz = horz.astype('uint8')
#     vert = vert.astype('uint8')

#     # Show the components as images
#     # cv2.imshow('Horizontal Component', horz)
#     cv2.imshow('Vertical Component', vert)

    

#     # Save with 's' or move on with delay
#     k = cv2.waitKey(500) & 0xff  # wait 500ms (half a second)
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite(f'opticalflow_frame{i}.png', vis)

#     prvs = next.copy()
#     frame_counter += 1           

# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import os

# # Folder containing the image sequence
# image_folder = 'output_files'

# # Get sorted list of image filenames
# image_files = sorted([
#     os.path.join(image_folder, f)
#     for f in os.listdir(image_folder)
#     if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
# ])

# # Make sure we have at least two frames
# if len(image_files) < 2:
#     print("Not enough images in folder to compute optical flow.")
#     exit()

# # Read the first image
# frame1 = cv2.imread(image_files[0])
# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# frame_counter = 1
# step = 16

# for i in range(1, len(image_files)):
#     # Read the next image
#     frame2 = cv2.imread(image_files[i])
#     next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     # Compute optical flow
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Extract only the vertical component
#     vert_flow = flow[..., 1]

#     # Draw vertical flow on the frame
#     vis = frame2.copy()
#     h, w = next.shape
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fy = vert_flow[y, x]
#             end_point = (x, int(y + fy))
#             cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

#     # Display vertical component image
#     cv2.imshow('Vertical Flow Visualization', vis)

#     # Optionally show the grayscale normalized version too
#     vert_norm = cv2.normalize(vert_flow, None, 0, 255, cv2.NORM_MINMAX)
#     vert_img = vert_norm.astype('uint8')
#     cv2.imshow('Vertical Component (Grayscale)', vert_img)

#     # Wait for key
#     k = cv2.waitKey(500) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite(f'verticalflow_frame{i}.png', vis)
#         cv2.imwrite(f'vertical_component_frame{i}.pgm', vert_img)

#     prvs = next.copy()
#     frame_counter += 1

# cv2.destroyAllWindows()

# show the vertical optical flow (farneback) on video -----------------------------------------
# import cv2
# import numpy as np
# import os

# # Folder containing the image sequence
# image_folder = 'out_directory_crop_comb'

# # Get sorted list of image filenames
# image_files = sorted([
#     os.path.join(image_folder, f)
#     for f in os.listdir(image_folder)
#     if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
# ])

# # Make sure we have at least two frames
# if len(image_files) < 2:
#     print("Not enough images in folder to compute optical flow.")
#     exit()

# # Read the first image
# frame1 = cv2.imread(image_files[0])
# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# frame_counter = 1
# step = 16

# for i in range(1, len(image_files)):
#     # Read the next image
#     frame2 = cv2.imread(image_files[i])
#     next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     # Compute optical flow
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Extract only the vertical component
#     vert_flow = flow[..., 1]

#     # Draw vertical flow on the frame
#     vis = frame2.copy()
#     h, w = next.shape
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fy = vert_flow[y, x]
#             end_point = (x, int(y + fy))
#             cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

#     # Display vertical component image
#     cv2.imshow('Vertical Flow Visualization', vis)

#     # Optionally show the grayscale normalized version too
#     vert_norm = cv2.normalize(vert_flow, None, 0, 255, cv2.NORM_MINMAX)
#     vert_img = vert_norm.astype('uint8')
#     cv2.imshow('Vertical Component (Grayscale)', vert_img)

#     # Wait for key
#     k = cv2.waitKey(500) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite(f'verticalflow_frame{i}.png', vis)
#         cv2.imwrite(f'vertical_component_frame{i}.pgm', vert_img)

#     prvs = next.copy()
#     frame_counter += 1

# cv2.destroyAllWindows()

# averge vertical motion magnitudes between frames ------------------------

# import cv2
# import numpy as np
# import os

# # Folder containing the image sequence
# image_folder = 'out_directory_crop_comb'

# # Get sorted list of image filenames
# image_files = sorted([
#     os.path.join(image_folder, f)
#     for f in os.listdir(image_folder)
#     if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')
# ])

# # Make sure we have at least two frames
# if len(image_files) < 2:
#     print("Not enough images in folder to compute optical flow.")
#     exit()

# # Read the first image
# frame1 = cv2.imread(image_files[0])
# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# frame_counter = 1
# step = 16

# # Optional: Store difference magnitudes
# vertical_motion_magnitudes = []

# for i in range(1, len(image_files)):
#     # Read the next image
#     frame2 = cv2.imread(image_files[i])
#     next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     # Compute optical flow
#     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # Extract only the vertical component
#     vert_flow = flow[..., 1]

#     # --- Calculate magnitude of vertical motion ---
#     vert_magnitude = np.mean(np.abs(vert_flow))
#     vertical_motion_magnitudes.append(vert_magnitude)
#     print(f"Frame {i-1} to {i}: Mean vertical motion = {vert_magnitude:.4f}")

#     # Draw vertical flow on the frame
#     vis = frame2.copy()
#     h, w = next.shape
#     for y in range(0, h, step):
#         for x in range(0, w, step):
#             fy = vert_flow[y, x]
#             end_point = (x, int(y + fy))
#             cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

#     # Display vertical component image
#     cv2.imshow('Vertical Flow Visualization', vis)

#     # # Optionally show the grayscale normalized version too
#     # vert_norm = cv2.normalize(vert_flow, None, 0, 255, cv2.NORM_MINMAX)
#     # vert_img = vert_norm.astype('uint8')
#     # cv2.imshow('Vertical Component (Grayscale)', vert_img)

#     # Wait for key
#     k = cv2.waitKey(500) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv2.imwrite(f'verticalflow_frame{i}.png', vis)
#         cv2.imwrite(f'vertical_component_frame{i}.pgm', vert_img)

#     prvs = next.copy()
#     frame_counter += 1

# cv2.destroyAllWindows()

# # Optionally print all stored magnitudes at the end
# print("\nAll vertical motion magnitudes between frames:")
# for idx, mag in enumerate(vertical_motion_magnitudes, start=1):
#     print(f"Frame {idx-1} to {idx}: {mag:.4f}")

# --------------------- region based ---------------------------------

import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt

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

# Store magnitudes and other metrics
vertical_motion_magnitudes = []
vertical_motion_std_devs = []
regionwise_std_devs = []

for i in range(1, len(image_files)):
    frame2 = cv2.imread(image_files[i])
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vert_flow = flow[..., 1]  # vertical component

    # --- Whole-frame analysis ---
    vert_magnitude = np.mean(np.abs(vert_flow))
    vert_std = np.std(vert_flow)

    vertical_motion_magnitudes.append(vert_magnitude)
    vertical_motion_std_devs.append(vert_std)

    print(f"Frame {i-1} to {i}:")
    print(f"  Mean vertical motion = {vert_magnitude:.4f}")
    print(f"  Std Dev (overall)    = {vert_std:.4f}")

    # --- Grid-based region analysis ---
    h, w = next.shape
    grid_size = 4  # e.g., 4x4 grid
    block_h = h // grid_size
    block_w = w // grid_size

    region_means = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            block = vert_flow[gy*block_h:(gy+1)*block_h, gx*block_w:(gx+1)*block_w]
            block_mean = np.mean(np.abs(block))
            region_means.append(block_mean)

    region_std = np.std(region_means)
    regionwise_std_devs.append(region_std)

    print(f"  Region-wise vertical motion means = {[round(m, 3) for m in region_means]}")
    print(f"  Region-wise std dev               = {region_std:.4f}")

    # --- Optional: Visualize vertical motion as arrows ---
    vis = frame2.copy()
    for y in range(0, h, step):
        for x in range(0, w, step):
            fy = vert_flow[y, x]
            end_point = (x, int(y + fy))
            cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.4)

    cv2.imshow('Vertical Flow Visualization', vis)

    # # --- Optional: Show heatmap ---
    # plt.imshow(np.abs(vert_flow), cmap='viridis')
    # plt.title(f"Vertical Motion Heatmap: Frame {i-1} to {i}")
    # plt.colorbar(label='Vertical Motion Magnitude')
    # plt.show()

    k = cv2.waitKey(500) & 0xff
    if k == 27:  # ESC
        break
    elif k == ord('s'):
        cv2.imwrite(f'verticalflow_frame{i}.png', vis)

    prvs = next.copy()
    frame_counter += 1

cv2.destroyAllWindows()

# --- Summary Output ---
print("\nSummary of Vertical Motion:")
for idx in range(len(vertical_motion_magnitudes)):
    print(f"Frame {idx} to {idx+1}:")
    print(f"  Mean    = {vertical_motion_magnitudes[idx]:.4f}")
    print(f"  Std Dev = {vertical_motion_std_devs[idx]:.4f}")
    print(f"  Region-wise Std Dev = {regionwise_std_devs[idx]:.4f}")







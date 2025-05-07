import cv2
import os

# Path to the input video
video_path = 'data/videos/example.mp4'  # Replace this with your actual video file path
output_folder = 'out_directory'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(f"Video FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")

# Frame interval for 1 frame per second
frame_interval = int(fps)

frame_idx = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        # Save frame as image
        frame_name = f"frame_{saved_count:04d}.png"
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")
        saved_count += 1

    frame_idx += 1

cap.release()
print(f"Done. {saved_count} frames saved to '{output_folder}'.")

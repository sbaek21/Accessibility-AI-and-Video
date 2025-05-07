import cv2
import os

def extract_and_save_scenes(video_path, output_dir):
    """
    Extracts one frame per second from the provided video, saves them as image files locally,
    and returns scenes as a dictionary.

    Parameters:
    video_path (string): Path of the video to be processed.
    output_dir (string): Directory where the extracted frames will be saved.

    Returns:
    list of dict: Each dictionary corresponds to a scene, containing:
        - index (int): Scene index
        - img_file (string): File name of the saved image
    """
    
    if not os.path.exists(video_path):
        print(f"{video_path}: File not found.")
        return []

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        print(f"Error: Could not retrieve FPS from {video_path}")
        return []

    scenes = []
    index = 0
    success, frame = cap.read()
    count = 0

    while success:
        # Capture frame every second (approximately)
        if count % int(fps) == 0:
            # Define the file name for the saved image
            img_file = os.path.join(output_dir, f"scene_{index}.jpg")
            
            # Save the frame as an image file
            cv2.imwrite(img_file, frame)
            
            # Append scene info to the list
            scenes.append({
                "index": index,
                "img_file": img_file
            })
            
            index += 1
        
        success, frame = cap.read()
        count += 1

    cap.release()
    
    return scenes

# Example Usage:
video_path = "data/videos/example.mp4" # Replace this with your actual video file path
output_dir = "out_directory_demo"
scenes = extract_and_save_scenes(video_path, output_dir)
print(f"Extracted and saved {len(scenes)} scenes.")

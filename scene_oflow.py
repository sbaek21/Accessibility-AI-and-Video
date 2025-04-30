import cv2
import numpy as np
import os
from mtcnn_cv2 import MTCNN

# Initialize the face detector (MTCNN)
detector = MTCNN()

def require_face_result(curr_frame):
    """
    Detects faces and upper bodies in the frame using MTCNN.
    
    Parameters:
        curr_frame (numpy.ndarray): The input frame (BGR).
    
    Returns:
        tuple: (has_person, boxes)
            - has_person (bool): True if a face (and optionally its upper body) is detected.
            - boxes (list): List of bounding boxes [x1, x2, y1, y2] for detected regions.
    """
    # Resize for faster detection
    resized_frame = cv2.resize(curr_frame, (320, 240))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    has_person = False
    boxes = []
    
    for face in faces:
        x, y, width, height = face['box']
        boxes.append([x, x + width, y, y + height])
        # A simple heuristic: if the face is large enough, consider that a person is present.
        if width / 320 > 0.1 or height / 240 > 0.1:
            has_person = True
    return has_person, boxes

def extract_non_scrolling_and_no_face_scenes(video_path, output_dir, scrolling_threshold=2.0):
    if not os.path.exists(video_path):
        print(f"{video_path}: File not found.")
        return []
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Error: Could not retrieve FPS from {video_path}")
        cap.release()
        return []
    
    # Setup optical flow parameters (using Shi-Tomasi for feature detection and Lucas-Kanade for tracking)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    scenes = []
    frame_count = 0
    scene_index = 0

    # Read the first frame and initialize feature points.
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read the video.")
        cap.release()
        return scenes

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow between the previous and current frame.
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
        scrolling = False
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            displacements = good_new - good_old
            avg_disp = np.mean(np.linalg.norm(displacements, axis=1))
            if avg_disp > scrolling_threshold:
                scrolling = True

        # Run face detection on the current frame.
        has_person, boxes = require_face_result(frame)
        
        # Only extract the frame if:
        # - It is not part of a scrolling period,
        # - No face (or upper-body) is detected,
        # - And it is at approximately one frame per second.
        if (not scrolling) and (not has_person) and (frame_count % int(fps) == 0):
            img_file = os.path.join(output_dir, f"scene_{scene_index}.jpg")
            cv2.imwrite(img_file, frame)
            scenes.append({
                "index": scene_index,
                "img_file": img_file
            })
            scene_index += 1

        # Update previous frame and features for optical flow.
        prev_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        frame_count += 1

    cap.release()
    return scenes

# Example usage:
video_path = "data/videos/test3_87s.mp4"
output_dir = "out_directory_no_face2"
scenes = extract_non_scrolling_and_no_face_scenes(video_path, output_dir, scrolling_threshold=2.0)
print(f"Extracted and saved {len(scenes)} scenes.")

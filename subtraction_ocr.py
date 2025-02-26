import cv2
import numpy as np
import os
import pytesseract
import re
# from Katna.video import Video

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Video file path
video_path = "test10_BIOE360_Chap14Part5 (1).mp4"

# Output directory
output_dir = "output_filtered_c"
os.makedirs(output_dir, exist_ok=True)

# Directory for keyframes w/ OCR
different_images_dir = os.path.join(output_dir, "different_images")
os.makedirs(different_images_dir, exist_ok=True)


# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
sampling_interval = 1  # Extract every 1 second

frame_index = 0
prev_text_chunks = set()
comparison_text_chunks = set()  # Stores the reference text chunks
different_image_indexes = []

# Function to extract text using OCR
def extract_text(frame):
    text = pytesseract.image_to_string(frame, config="--psm 6").strip()
    return text.replace('\n', ' ').replace('\x0c', '').strip()

# Function to split text into ordered word chunks
def get_word_chunks(text, chunk_size=4):
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words
    chunks = {' '.join(words[i:i+chunk_size]) for i in range(len(words) - chunk_size + 1)}
    return chunks

# Function to check similarity using Jaccard index
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0  # If either is empty, assume no similarity
    return len(set1 & set2) / len(set1 | set2)

# Extract frames
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index * fps * sampling_interval))
    ret, frame = cap.read()

    if not ret:
        break  # End of video

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract text using OCR
    extracted_text = extract_text(gray_frame)

    # Ignore frames where text is too short (reduce OCR noise)
    if len(extracted_text.split()) < 10:
        frame_index += 1
        continue

    # Generate word chunks
    new_text_chunks = get_word_chunks(extracted_text, chunk_size=4)

    # Compare similarity with comparison frame
    comparison_similarity = jaccard_similarity(comparison_text_chunks, new_text_chunks)

    # Compare similarity with previous frame
    text_changed = jaccard_similarity(prev_text_chunks, new_text_chunks) < 0.15  # Less than 15% overlap

    # If **40% or less similar to first frame**, update comparison frame
    if comparison_similarity <= 0.40:
        comparison_text_chunks = new_text_chunks
        print(f"Updated comparison frame to frame {frame_index}")

    # Save frame **only if the text is substantially different**
    if text_changed:
        frame_path = os.path.join(different_images_dir, f"frame_{frame_index}.jpg")
        cv2.imwrite(frame_path, gray_frame)
        different_image_indexes.append(frame_index)
        print(f"Saved frame {frame_index} due to significant text change")

        prev_text_chunks = new_text_chunks  # Update previous text chunks

    frame_index += 1

cap.release()
print("Frames with significant differences:", different_image_indexes)

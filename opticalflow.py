# import numpy as np
# import cv2 as cv

# cap = cv.VideoCapture(0)  # Use default webcam
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# # params for ShiTomasi corner detection
# feature_params = dict(maxCorners=100,
#                       qualityLevel=0.3,
#                       minDistance=7,
#                       blockSize=7)

# # Parameters for Lucas-Kanade optical flow
# lk_params = dict(winSize=(15, 15),
#                  maxLevel=2,
#                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# # Create some random colors
# color = np.random.randint(0, 255, (100, 3))

# # Capture the first frame and flip it horizontally
# ret, old_frame = cap.read()
# if not ret:
#     print("Error: Cannot capture frame from webcam.")
#     cap.release()
#     cv.destroyAllWindows()
#     exit()

# # Flip the frame to avoid mirror view
# old_frame = cv.flip(old_frame, 1)
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print('No frames grabbed!')
#         break

#     # Flip the current frame to avoid mirror view
#     frame = cv.flip(frame, 1)
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # Calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     # Select good points
#     if p1 is not None:
#         good_new = p1[st == 1]
#         good_old = p0[st == 1]

#         # Draw the tracks
#         for i, (new, old) in enumerate(zip(good_new, good_old)):
#             a, b = new.ravel()
#             c, d = old.ravel()
#             mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
#             frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

#     img = cv.add(frame, mask)
#     cv.imshow('Optical Flow', img)
#     if cv.waitKey(30) & 0xFF == 27:
#         break

#     # Update previous frame and points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)

# cap.release()
# cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv

# cap = cv.VideoCapture(0)  # Use default webcam
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# # Params for ShiTomasi corner detection
# feature_params = dict(maxCorners=100,
#                       qualityLevel=0.3,
#                       minDistance=7,
#                       blockSize=7)

# # Parameters for Lucas-Kanade optical flow
# lk_params = dict(winSize=(15, 15),
#                  maxLevel=2,
#                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# # Create some random colors
# color = np.random.randint(0, 255, (100, 3))

# # Capture the first frame and flip it horizontally
# ret, old_frame = cap.read()
# if not ret:
#     print("Error: Cannot capture frame from webcam.")
#     cap.release()
#     cv.destroyAllWindows()
#     exit()

# # Flip the frame to avoid mirror view
# old_frame = cv.flip(old_frame, 1)
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print('No frames grabbed!')
#         break

#     # Flip the current frame to avoid mirror view
#     frame = cv.flip(frame, 1)
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # Calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     # Select good points
#     if p1 is not None:
#         good_new = p1[st == 1]
#         good_old = p0[st == 1]

#         # Draw the tracks
#         for i, (new, old) in enumerate(zip(good_new, good_old)):
#             a, b = new.ravel()
#             c, d = old.ravel()
#             mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
#             frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

#     # Combine frame with the mask and add demo text
#     img = cv.add(frame, mask)
#     cv.putText(img, 'Optical Flow Demo', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv.imshow('Optical Flow', img)

#     if cv.waitKey(30) & 0xFF == 27:
#         break

#     # Update previous frame and points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)

# cap.release()
# cv.destroyAllWindows()








####USING WEBCAM


import numpy as np
import cv2 as cv

# Open the default webcam and set the capture resolution.
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Create a resizable window and set its size to a larger display.
cv.namedWindow('Optical Flow Demo', cv.WINDOW_NORMAL)
cv.resizeWindow('Optical Flow Demo', 1280, 960)

# Parameters for ShiTomasi corner detection.
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow.
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create an array of random colors.
color = np.random.randint(0, 255, (100, 3))

# Capture the first frame, flip it horizontally, and detect features.
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot capture frame from webcam.")
    cap.release()
    cv.destroyAllWindows()
    exit()

old_frame = cv.flip(old_frame, 1)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing optical flow tracks.
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # Flip frame for non-mirrored view.
    frame = cv.flip(frame, 1)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method.
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # If flow is found, select good points and draw the tracks.
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    else:
        # If tracking fails, you might want to reinitialize tracking.
        good_new = p0

    # Combine the current frame with the drawn tracks.
    img = cv.add(frame, mask)

    # Overlay demo instructions.
    cv.putText(img, "Optical Flow Demo - Press 'r' to reset, 'ESC' to exit", 
               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the image in the larger window.
    cv.imshow('Optical Flow Demo', img)

    key = cv.waitKey(30) & 0xFF
    if key == 27:  # ESC key to exit.
        break
    elif key == ord('r'):  # Reset tracking when 'r' is pressed.
        # Reinitialize tracking on the current frame.
        old_frame = frame.copy()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)
        continue  # Skip updating the old frame below for this cycle.

    # Update for the next iteration.
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()




import numpy as np
import cv2 as cv

# Use a video file from your laptop instead of the webcam.
cap = cv.VideoCapture("data/videos/Stat Video.mp4")  # Replace "video.mp4" with your video file path

# Optionally set resolution (if needed) depending on your video.
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Create a resizable window and set its size to a larger display.
cv.namedWindow('Optical Flow Demo', cv.WINDOW_NORMAL)
cv.resizeWindow('Optical Flow Demo', 1280, 960)

# Parameters for ShiTomasi corner detection.
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow.
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create an array of random colors.
color = np.random.randint(0, 255, (100, 3))

# Capture the first frame and detect features.
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot capture frame from video file.")
    cap.release()
    cv.destroyAllWindows()
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing optical flow tracks.
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed or end of video reached!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method.
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # If flow is found, select good points and draw the tracks.
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    else:
        # If tracking fails, maintain the previous points.
        good_new = p0

    # Combine the current frame with the drawn tracks.
    img = cv.add(frame, mask)

    # Overlay demo instructions.
    cv.putText(img, "Optical Flow Demo - Press 'r' to reset, 'ESC' to exit", 
               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the image in the larger window.
    cv.imshow('Optical Flow Demo', img)

    key = cv.waitKey(30) & 0xFF
    if key == 27:  # ESC key to exit.
        break
    elif key == ord('r'):  # Reset tracking when 'r' is pressed.
        # Reinitialize tracking on the current frame.
        old_frame = frame.copy()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)
        continue  # Skip updating the old frame below for this cycle.

    # Update for the next iteration.
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()


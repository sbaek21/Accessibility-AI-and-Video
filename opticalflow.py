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


import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)  # Use default webcam
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Capture the first frame and flip it horizontally
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot capture frame from webcam.")
    cap.release()
    cv.destroyAllWindows()
    exit()

# Flip the frame to avoid mirror view
old_frame = cv.flip(old_frame, 1)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # Flip the current frame to avoid mirror view
    frame = cv.flip(frame, 1)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    # Combine frame with the mask and add demo text
    img = cv.add(frame, mask)
    cv.putText(img, 'Optical Flow Demo', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow('Optical Flow', img)

    if cv.waitKey(30) & 0xFF == 27:
        break

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()

# Handling Annotations

Given an engineering video lecture, aims to distinguish frames with annotations and redundant frames from scene changes.

## Example pipeline

![masking_subtraction.png](/images/masking_subtraction.png)

The pixels remaining on the subtracted frames are then counted, determining where the scene changes are located.

## Masking
We used Thresholding to mask our segmented frames.

This works by applying a Gaussian Blur on selected frames to remove noise, then limiting the total amount of rgb values on the frame. 

Then thresholding is applied where the entire image is converted into black and white based on a constant threshold $(a_{x,y} < t)$.

Original Image:
![original.jpg](/images/original.jpg)

Thresholding to extract foreground:
![foreground.png](/images/foreground.png)

Masked image:
![masked.jpg](/images/masked.jpg)


## Subtraction
"Subtracts" the frames across a video.  For each pair of adjacent frames, frame1 and frame2, checks every pixel at (x,y).  If frame1(x,y) and frame2(x,y) have the same pixel within a hardcoded threshold, frame1(x,y) is set to null.  Otherwise, frame1(x,y) is left unchanged.  This is repeated for every adjacent pair of frames (frame2, frame3), (frame3, frame4), and so on so forth.

Next, the number of non-null pixels (remaining pixels) are counted for each frame.  Then, if a frame has >1.5% of pixels remaining, the frame is predicted to have a scene change.

## Masking and subtraction
Combines the above files, updates frame extraction, speeds up subtraction and remaining pixel counting, uses data to create a prediction of whether a frame is a scene change, and outputs results, plus some other small changes.

Note: does not filter out mid-scroll frames, leading to some false positives.

### Example output
Extracting frames 19:47:37

Creating masks 19:47:42

Starting subtraction 19:48:19

Counting remaining 19:48:41

Selecting frames 19:48:41

[21, 25, 226, 227, 228, 282, 300, 301, 320, 321, 354, 355, 356, 369, 370, 371, 385, 386, 407, 408, 409, 410, 411, 412, 413, 547, 548, 549, 550, 551, 552]

Finished 19:48:41
# CNN


# Labeling
A small notebook to help label frames of a video as scene changes/not scene changes with a flexible interval between selected frames (sampling frequency).  Traverses given video backwards.  Displays two buttons ('scene change' and 'no scene change') and two frames from the video.  The top frame chronologically preceeds the bottom frame by one interval.  

If 'scene change' is clicked, a row with the frame number, sample number (0-indexed), and 'True' is written to the csv.  

If 'scene change' is clicked, a row with the frame number, sample number, and 'false' is written.  
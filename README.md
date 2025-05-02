# Software_project2
Object recognition using deep learning for ECEN 4273 - Recognizes cat, dog, human, lightsaber, and dalek
Brent Bertaux, Katilynn Mar, Bailey Chave

Library dependencies:
    opencv-python
    torch
    ultralytics
    pytest

Installation & Usage Procedures: (after downloading the git repository)
There are two programs included in the final submission:
    liveFeed.py: to detect objects in real time
        To use: Ensure the webcam is not being used elsewhere, then run as-is. Webcam will open on your laptop, and a popup window will show recognition in real time. 
    objectDetectionWithVideo.py: to process a video upload and output a copy with the bounding boxes
        To use: Download video file you want to analyze, then update line 61 with the file path to the video. You can change the output file name on Line 41. Run, then view the output file.


Documentation was automatically developed using Pydoc. The pydoc file is labeled 'objectDoc.py' which creates 'objectDoc.html'

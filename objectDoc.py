"""
Object Detection Program using YOLOv8

This program uses the YOLOv8 object detection model (from the ultralytics package)
to detect specified objects in a video file. It processes the video frame by frame,
draws bounding boxes and labels on detected objects, and saves the output as a new video.

Supported detections include: person, cat, dog, dalek, and lightsaber.

Dependencies:
- OpenCV (cv2)
- ultralytics (YOLO)

Author: Your Name
Date: YYYY-MM-DD
"""

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the pre-trained YOLOv8 model

# Define the object names to detect
OBJECTS_TO_DETECT = {"person", "cat", "dog", "dalek", "lightsaber"}


def detect_objects(frame):
    """
    Detect and annotate specified objects in a video frame.

    Args:
        frame (numpy.ndarray): The video frame to process.

    Returns:
        numpy.ndarray: The frame with bounding boxes and labels drawn
        for detected objects.
    
    Note:
        Only objects listed in OBJECTS_TO_DETECT will be annotated.
    """
    results = model(frame)  # Run detection
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            
            if label in OBJECTS_TO_DETECT:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def process_video(video_path):
    """
    Process a video file, detecting and annotating specified objects.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        None

    Raises:
        IOError: If the video file cannot be opened.
    
    Note:
        The processed video is saved as 'sillyKitty.avi' in the current directory.
        Press 'q' during playback to quit early.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_video = cv2.VideoWriter('sillyKitty.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_objects(frame)
        output_video.write(frame)  # Save the frame with detections
        
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\berta\Downloads\9BA9D88F-A31D-4F27-A84A-174F6C4CB2BB.MP4"
    process_video(video_path)

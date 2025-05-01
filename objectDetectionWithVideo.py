import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the pre-trained YOLOv8 model

# Define the object names to detect
OBJECTS_TO_DETECT = {"person", "cat", "dog", "dalek", "lightsaber"}

# Function to detect objects in a frame
def detect_objects(frame):
    results = model(frame)  # Run detection
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            
            if label in OBJECTS_TO_DETECT:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Video processing function
def process_video(video_path):
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

# Run the detection on the specified video file
if __name__ == "__main__":
    video_path = r"C:\Users\berta\Downloads\9BA9D88F-A31D-4F27-A84A-174F6C4CB2BB.MP4"  # Raw string to handle backslashes
    process_video(video_path)
# "C:\Users\berta\Downloads\filtered-B66D5243-35AA-4AE3-8AE0-1A4EDDBD7D4D.MP4"
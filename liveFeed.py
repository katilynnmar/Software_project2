import cv2
import torch
from ultralytics import YOLO

# ================== Model Loading ===================

# Automatically select device: GPU (cuda:0) if available, else CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load default YOLOv8n model for person, cat, dog
default_model = YOLO("yolov8n.pt")
default_model.to(device)

# Load your Dalek model
dalek_model = YOLO(r"C:\Users\berta\Downloads\dalek_lightsaber\dalek\weights\best.pt")


dalek_model.to(device)

# Load your Lightsaber model
lightsaber_model = YOLO(r"C:\Users\berta\Downloads\dalek_lightsaber\lightsaber\weights\best.pt")
lightsaber_model.to(device)

# ================== Label Filtering ===================

DEFAULT_OBJECTS = {"person", "cat", "dog"}
CUSTOM_OBJECTS = {"Dalek", "Dalek", "sith light", "jedi lightsaber"}

# ================== Detection Functions ===================

def draw_boxes(results, frame, model, allowed_labels):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            if label in allowed_labels:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detect_objects(frame):
    results_default = default_model.predict(frame, device=device, verbose=False)
    results_dalek = dalek_model.predict(frame, device=device, verbose=False)
    results_lightsaber = lightsaber_model.predict(frame, device=device, verbose=False)

    draw_boxes(results_default, frame, default_model, DEFAULT_OBJECTS)
    draw_boxes(results_dalek, frame, dalek_model, CUSTOM_OBJECTS)
    draw_boxes(results_lightsaber, frame, lightsaber_model, CUSTOM_OBJECTS)

    return frame

# ================== Video Streaming ===================

def process_video(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)
        cv2.imshow("Live Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================== Main Run ===================

if __name__ == "__main__":
    process_video(0)
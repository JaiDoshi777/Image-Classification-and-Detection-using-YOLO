from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import torch

# cap = cv.VideoCapture(0) # For webcam
# cap.set(3, 1280) # propperty ID: 3 = frame width    in case of video path reading u cant set a frame size,,
# cap.set(4, 720) # propperty ID: 4 = frame height

cap = cv.VideoCapture("D:/Computer Vision Udemy Course/External_projects/pythonProject/videos/people.mp4") # to capture from path..

model = YOLO('yolo_weights/yolov8n.pt').to('cuda')

device = model.device  # Check device used by the model
print(f"Model is using device: {device}")
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
device = model.device  # Check device used by the model
print(f"Model is using device: {device}")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Video Propertioes for saving
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

# Defining the CODEC and create VideoWriter object


fourcc = cv.VideoWriter_fourcc(*'mp4v') # Codec for mp4 file
out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame or end of video.")
        break

    # Ensure frame dimensions are valid
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print("Empty frame detected.")
        break

    results = model.predict(source=frame, stream=True)
    for r in results:
        boxes = r.boxes # A property of the result r that contains all the bounding boxes
        # predictions for objects detected in the frame.
        for box in boxes:

            # BOUNDING BOX

            x1,y1,x2,y2 = box.xyxy[0] # Provides the bounding box coordinates in the
            # [x1, y1, x2, y2] format, [0] - even if the box.xyxy 2D array the 0 makes sure that
            # we only take the contents of the inner 1st 1D array,, if [[x,y,x,y,]]
            # then we take only [x,y,x,y,], which is needed for backend purpose
            x,y,w,h = x1,y1,x2,y2
            x1, y1, x2, y2 = map(int, [x1,y1,x2,y2]) # int conversion required for opencv

            # Drawing rectagle with openCV
            # cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

            # Drawing rectangle with cvzone
            cvzone.cornerRect(frame,bbox=(x1,y1,x2-x1,y2-y1))


            # CONFIDENCE SCORE

            score = math.ceil((box.conf[0]*100))/100 # conf - short form of confidence score,,
            # The correct attribute to access the confidence score of a bounding box is box.conf


            # CLASS NAMES

            cls = int(box.cls[0]) # short form of class of the perticular detection, like the confidence score
            cvzone.putTextRect(frame, f'{classNames[cls]} {score}', (max(0, x1), max(40, y1)), scale=1, thickness=1)

    # Saving processed frame
    out.write(frame)

    cv.imshow('IMAGE', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break


out.release()
cap.release()
cv.destroyAllWindows()
# Image-Classification-and-Detection-using-YOLO
This project is designed to identify and classify objects within an image using the YOLO (You Only Look Once) model. 

The script begins by importing essential libraries, including cv2 for OpenCV functionalities, numpy for numerical operations, time for time-related functions, and os for operating system interactions. The working directory is set to the location of the YOLO model files, and the necessary files are listed.

A custom YOLO model is imported from the model.updated_yolo_model module. The process_image function is defined to preprocess the input image by resizing it to 416x416 pixels, normalizing the pixel values, and expanding the dimensions to match the input shape expected by the YOLO model.

The get_classes function reads the class names from a specified file (coco_classes.txt) and returns a list of class names. This file contains the names of the objects that the YOLO model can detect.

The draw function is responsible for drawing bounding boxes and class labels on the detected objects in the image. For each detected object, the bounding box coordinates (x, y, w, h), confidence score, and class label are extracted. A rectangle is drawn around the object using cv.rectangle, and the class label and confidence score are displayed using cv.putText.

The detect_image function processes the input image using the process_image function and performs object detection using the YOLO model. The yolo.predict method returns the bounding boxes, class labels, and confidence scores for the detected objects. The detection time is printed for reference. If any objects are detected, the draw function is called to annotate the image with the detection results.

The YOLO model is initialized with confidence and IoU thresholds of 0.6 and 0.5, respectively. The class names are loaded from the coco_classes.txt file. An input image (test_image13.jpg) is read using cv.imread, and the detect_image function is called to perform object detection. The annotated image is saved to the specified output path using cv.imwrite.

![image](https://github.com/user-attachments/assets/63801fb8-a961-47f8-94c1-be69cad090b9)
![image](https://github.com/user-attachments/assets/35a8d516-0f09-482b-813a-a4ce002b2165)
![image](https://github.com/user-attachments/assets/d71a4a7d-d988-44d7-ad03-f8de9ec4a3f1)
![image](https://github.com/user-attachments/assets/e64df9ac-d37f-49c8-9c23-525d7acf7d67)
![image](https://github.com/user-attachments/assets/64a17b7f-fb72-416d-9c9c-cafcf9009bae)

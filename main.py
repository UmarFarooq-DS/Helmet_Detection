import os
import cv2
import numpy as np


cfg_path = os.path.join(os.getcwd(), "Model_Files", "yolov3-helmet.cfg")
weights_path = os.path.join(os.getcwd(), "Model_Files", "yolov3-helmet.weights")
names_path = os.path.join(os.getcwd(), "Model_Files", "helmet.names")



# Load class names
classes = []
with open(names_path, "r") as f:
  classes = [line.strip() for line in f.readlines()]

# Load the YOLOv3 model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Function to detect objects in an image
def detect_objects(img):
  # Create a blob from the image
  blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
  
  # Set the input to the network
  net.setInput(blob)

  # Get the output layer names
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

  # Forward pass
  outs = net.forward(output_layers)

  #  Thresholding and Non-maxima Suppression (NMS) - You can adjust these values
  conf_threshold = 0.5
  nms_threshold = 0.4

  # Initialize lists for bounding boxes and confidences
  boxes = []
  confidences = []

  # Loop through each output layer
  for out in outs:
    for detection in out:
      scores = detection[5:]  # Confidence scores for each class
      class_id = np.argmax(scores)  # Get the index of the class with the highest score
      confidence = scores[class_id]
      
      # Filter based on confidence threshold
      if confidence > conf_threshold:
        # Scale bounding box coordinates to original image size
        center_x = int(detection[0] * img.shape[1])
        center_y = int(detection[1] * img.shape[0])
        w = int(detection[2] * img.shape[1])
        h = int(detection[3] * img.shape[0])
        x = center_x - w // 2
        y = center_y - h // 2
        
        # Append to lists
        boxes.append([x, y, w, h])
        confidences.append(confidence)

  # Apply non-maxima suppression
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

  # Draw bounding boxes and labels on the image
  if len(idxs) > 0:
    for i in idxs.flatten():
      x, y, w, h = boxes[i]
      color = (255, 0, 0)  # Change color for bounding boxes (BGR)
      cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
      text = f"{classes[class_id]}: {confidences[i]:.2f}"
      cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  return img

# input proecessing (input is either camera, an image or a video)
def process_input(input_path):
    # For realtime or camera detection
    if (input_path == "0"):
      cap = cv2.VideoCapture(0)
      while True:
        ret, frame = cap.read()

        if ret:
            # Detect objects in the framed
            result_frame = detect_objects(frame)
            
            # Display the result
            cv2.imshow("Video with Camera", result_frame)

        if cv2.waitKey(0):
            break
        
      cap.release()
      cv2.destroyAllWindows()


    # Determine if the input is an image or a video
    elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
      # Process image
      image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
      
      # Check if the image has 4 channels (RGBA)
      if image.shape[2] == 4:
          # Convert the image from RGBA to RGB
          image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
      
      # Detect objects in the image
      result_frame = detect_objects(image)

      # Display the result
      cv2.imshow("Image with Detection", result_frame)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    
    else:
      # Process video
      cap = cv2.VideoCapture(input_path)
      
      while True:
        ret, frame = cap.read()
        if not ret:
          break

        # Detect objects in the framed
        result_frame = detect_objects(frame)
        
        # Display the result
        cv2.imshow("Video with Detection", result_frame)

        if cv2.waitKey(0):
          break
      
      cap.release()
      cv2.destroyAllWindows()

img = input("Enter an image/video path OR 0 for camera: ")
img = os.fspath(img)

process_input(img) # pass 0 for helmet detection using camera

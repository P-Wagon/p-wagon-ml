import cv2
import pickle
import numpy as np

# Load the trained scaler and SVC from the .pkl file
with open('X_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('svc.pkl', 'rb') as f:
    svc = pickle.load(f)

# Function to preprocess the image and extract features
def preprocess_image(image, scaler):
    # Assuming the image needs to be scaled to the same size as the training images
    # And assuming feature extraction similar to the training process is needed
    # This needs to be adapted to your specific preprocessing and feature extraction steps
    
    # For example, resize image if necessary
    resized_image = cv2.resize(image, (64, 64))  # Example size, adjust to your needs
    
    height, width, channels = resized_image.shape
    image_flattened = resized_image.reshape((height * width, channels))

    #   Now scale the flattened image data
    scaled_features = scaler.transform(image_flattened)

# After processing, if you need to convert it back to the original shape for some reason:
    image_processed = scaled_features.reshape((height, width, channels))
    scaled_features = scaler.transform([resized_image])  # The input to transform should be an array-like object
    
    return scaled_features

# Function to perform object detection
def detect_objects(frame, scaler, svc):
    # Preprocess the image and extract features
    features = preprocess_image(frame, scaler)
    
    # Perform prediction using the trained SVC
    prediction = svc.predict(features)
    
    return prediction

# OpenCV to read the video stream or an image
cap = cv2.VideoCapture(0)  # Use 0 for webcam, replace with video file path for video

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Object detection
    prediction = detect_objects(frame, scaler, svc)
    
    # If an object is detected, draw a bounding box or other annotation
    # (This part will depend on the output of your SVC and how you want to annotate detections)
    if prediction == 1:  # Assuming '1' means object detected
        # Draw bounding box (example coordinates, you need to determine these based on your SVC output)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
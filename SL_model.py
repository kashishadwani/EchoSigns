import json
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Load trained model
model = tf.keras.models.load_model(r"C:/Users/User/minor_project/sign_language_model.h5")

# Load class indices from JSON file
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Get the label list from class indices
labels = list(class_indices)
print("Labels:", labels)

# Start webcam
cap = cv2.VideoCapture(0)
prediction_queue = deque(maxlen=5)  # Smoothing queue


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display_frame = frame.copy()

    # Preprocess frame
    processed_frame = cv2.flip(frame, 1)  # Flip horizontally
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Reduce noise
    threshold_frame = cv2.adaptiveThreshold(
        gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )  # Improve contrast
    img = cv2.resize(frame, (128, 128)) / 255.0  # Normalize
    # Fix: Convert image to uint8 before applying cv2.cvtColor()
    img = (img * 255).astype(np.uint8)

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.reshape(img, (1, 128, 128, 3))  # Ensure the final shape is correct



    # Make predictions
    pred = model.predict(img)
    print(f"Prediction Raw Output: {pred}")  # Debugging step
    print(f"Predicted Label Index: {np.argmax(pred)}")
    # predicted_label = labels[np.argmax(pred)]
    confidence_threshold = 0.7
    if np.max(pred) > confidence_threshold:
       predicted_label = labels[np.argmax(pred)]
    else:
       predicted_label = "Uncertain"


    # # Extract country and letter
    # predicted_country, predicted_letter = predicted_label.split("_")
    # predicted_country, predicted_letter = predicted_label.split("_")
    # Safely extract country and letter
    # smoothed_prediction = max(set(prediction_queue), key=prediction_queue.count)
    split_label = predicted_label.split("/")
    if len(split_label) == 2:
        predicted_country, predicted_letter = split_label
    else:
        predicted_country, predicted_letter = "Unknown", predicted_label  # Fallback for unexpected formats


    # Display result
    cv2.putText(frame, f"Letter: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Country: {predicted_country}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

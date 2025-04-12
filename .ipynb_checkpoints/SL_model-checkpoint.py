import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Load label mappings
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Modify if including digits
countries = ["America", "Filipino", "India", "Indonesia", "Malaysia"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    letter_pred, country_pred = model.predict(img)

    # Get predicted labels
    predicted_letter = letters[np.argmax(letter_pred)]
    predicted_country = countries[np.argmax(country_pred)]

    # Display result
    cv2.putText(frame, f"Letter: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Country: {predicted_country}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

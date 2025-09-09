import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load trained MNIST model
model = load_model("mnist_cnn_augmented.h5")

# Start webcam
cap = cv2.VideoCapture(0)

# For smoothing predictions
pred_queue = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Take the largest contour (assume it’s the digit)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Only consider reasonably large contours
        if w > 5 and h > 5:
            digit_roi = thresh[y:y+h, x:x+w]
            
            # Make square
            size = max(w, h)
            square = np.zeros((size, size), dtype=np.uint8)
            # Center the digit in the square
            x_offset = (size - w) // 2
            y_offset = (size - h) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi
            
            # Resize to 28x28
            digit_img = cv2.resize(square, (28, 28))
            
            # Normalize and reshape
            digit_img = digit_img.astype("float32") / 255.0
            digit_img = digit_img.reshape(1, 28, 28, 1)
            
            # Predict
            pred = model.predict(digit_img)
            predicted_digit = np.argmax(pred)
            
            # Add to queue for smoothing
            pred_queue.append(predicted_digit)
            smoothed_pred = max(set(pred_queue), key=pred_queue.count)
            
            # Draw rectangle and prediction
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Predicted: {smoothed_pred}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    cv2.imshow("Digit Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

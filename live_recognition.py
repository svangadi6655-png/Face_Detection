import os
import cv2
import numpy as np
import time
from face_recognition import load_dataset, EigenfacesPCA, MultilayerPerceptron

# Configuration
K_EIGENFACES = 80
CONFIDENCE_THRESHOLD = 0.55
IMG_SIZE = (64, 64)
DATASET_PATH = 'dataset/dataset/faces'

def main():
    # 1. Load the training dataset
    X_train, y_train, _, _, name_to_label = load_dataset(DATASET_PATH, img_size=IMG_SIZE)
    
    # Invert dictionary to get label -> name mapping
    label_to_name = {idx: name for name, idx in name_to_label.items()}
    
    print("\nTraining PCA & ANN model for live recognition...")
    
    # 2. Fit PCA model
    pca = EigenfacesPCA(k=K_EIGENFACES)
    train_signatures = pca.fit(X_train)
    
    # Calculate standardization parameters
    mean_sig = np.mean(train_signatures, axis=0, keepdims=True)
    std_sig = np.std(train_signatures, axis=0, keepdims=True)
    std_sig[std_sig == 0] = 1e-15
    train_signatures_norm = (train_signatures - mean_sig) / std_sig
    
    # 3. Train the Custom ANN
    num_classes = len(name_to_label)
    ann = MultilayerPerceptron(
        input_dim=train_signatures_norm.shape[1], 
        hidden_dim=128, 
        output_dim=num_classes, 
        lr=0.03, 
        reg=0.01
    )
    ann.train(train_signatures_norm, y_train, epochs=800, batch_size=32)
    print("Model trained successfully!")
    
    # 4. Initialize Haar Cascade for Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 5. Start Camera Capture (tries index 0 and 1, with DirectShow fallback for Windows)
    cap = None
    for index in [0, 1]:
        print(f"Trying to open webcam at index {index}...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Webcam successfully opened at index {index} using DirectShow backend.")
            break
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam successfully opened at index {index} using default backend.")
            break
            
    if cap is None or not cap.isOpened():
        print("Error: Could not access any webcam (tried index 0 and 1 with all backends).")
        print("Please check if your camera is connected or being used by another application.")
        return
        
    print("\n" + "=" * 50)
    print("Live Face Recognition Active!")
    print("Safety Feature:")
    print(" - The camera will automatically close after 15 seconds.")
    print(" - It will exit instantly once it detects a face stably for 15 frames.")
    print(" - Press 'q' key to quit manually.")
    print("=" * 50)
    
    # Initialize timing and stability variables
    start_time = time.time()
    consecutive_frames = 0
    last_pred_name = None
    
    while True:
        # Check 15-second timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > 15.0:
            print("\nAuthentication Timeout: No stable face detected within 15 seconds.")
            break
            
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=5, 
            minSize=(80, 80)
        )
        
        detected_name = None
        
        # Process the first detected face (for single-user authentication)
        for (x, y, w, h) in faces:
            # Crop the detected face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess the cropped face (resize and normalize)
            face_resized = cv2.resize(face_roi, IMG_SIZE).astype(np.float32) / 255.0
            face_flat = face_resized.flatten().reshape(-1, 1) # Shape: (mn, 1)
            
            # Project onto PCA Space
            face_sig = pca.transform(face_flat) # Shape: (1, k)
            
            # Standardize feature
            face_sig_norm = (face_sig - mean_sig) / std_sig
            
            # Predict identity using trained ANN
            probs = ann.forward(face_sig_norm)
            max_prob = np.max(probs)
            pred_class = np.argmax(probs)
            
            # Check confidence threshold for imposter detection
            if max_prob >= CONFIDENCE_THRESHOLD:
                name = label_to_name[pred_class]
                color = (0, 255, 0) # Green for recognized enrolled face
            else:
                name = "Unknown"
                color = (0, 0, 255) # Red for unknown face / imposter
                
            detected_name = name
            
            # Draw rectangle and label around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({max_prob * 100:.1f}%)"
            cv2.putText(
                frame, 
                label, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2, 
                cv2.LINE_AA
            )
            # Break to only authenticate one person at a time
            break
            
        # Display the resulting frame
        cv2.imshow('Live PCA+ANN Face Recognition', frame)
        
        # Stability / Auto-exit logic
        if detected_name is not None:
            if detected_name == last_pred_name:
                consecutive_frames += 1
            else:
                consecutive_frames = 1
                last_pred_name = detected_name
            
            # If recognized stably for 15 frames (~0.5 - 0.75 seconds of video feed)
            if consecutive_frames >= 15:
                if detected_name == "Unknown":
                    print(f"\nAccess Denied: Imposter/Unknown face detected stably!")
                else:
                    print(f"\nAccess Granted: Welcome {detected_name}! Identification verified.")
                break
        else:
            # If no face is detected in the frame, reset stability counter
            consecutive_frames = 0
            last_pred_name = None
            
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nWebcam session terminated manually.")
            break
            
    # Release camera and clean up windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam session closed.")

if __name__ == '__main__':
    main()

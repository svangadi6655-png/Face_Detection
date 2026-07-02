import os
import cv2
import time

def register():
    # Ask for user's name
    name = input("Enter your name to enroll: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
        
    save_dir = f"dataset/dataset/faces/{name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Fallback to index 1 or directshow if index 0 fails
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
                
    print("\n" + "=" * 50)
    print(f"Enrolling face for '{name}'...")
    print("Look at the camera. We will capture 50 images.")
    print("Move your head slightly to capture different angles.")
    print("=" * 50)
    print("Press 'q' at any time to cancel.")
    
    count = 0
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        
        # Draw target rect to show user where to look
        for (x, y, w, h) in faces:
            # Crop face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Draw green rectangle on visual frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing: {count}/50", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save cropped face to dataset folder
            img_name = f"{save_dir}/face_{100 + count}.jpg"
            # Resize cropped face to standard dataset size 100x100 (which will be resized to 64x64 during loading)
            face_resized = cv2.resize(face_roi, (100, 100))
            cv2.imwrite(img_name, face_resized)
            count += 1
            
            # Small delay between captures
            time.sleep(0.1)
            break # Only process one face at a time
            
        cv2.imshow("Register Face - Look Here", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    if count == 50:
        print(f"\nSuccessfully enrolled {name}! 50 face images saved to '{save_dir}'.")
        print("\nNext step: Add your name to the 'enrolled_names' list in: ")
        print("1. face_recognition.py")
        print("2. live_recognition.py")

if __name__ == '__main__':
    register()

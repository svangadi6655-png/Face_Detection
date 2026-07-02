"""
model_manager.py
Singleton that holds the trained PCA+ANN model.
Supports background retraining when new users are registered.
"""
import os
import threading
import numpy as np

# Import from the existing face_recognition module
from face_recognition import load_dataset, EigenfacesPCA, MultilayerPerceptron

DATASET_PATH = "dataset/dataset/faces"
IMG_SIZE = (64, 64)
K_EIGENFACES = 80
CONFIDENCE_THRESHOLD = 0.55
HIDDEN_DIM = 128
EPOCHS = 600
BATCH_SIZE = 32


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._pca = None
        self._ann = None
        self._mean_sig = None
        self._std_sig = None
        self._label_to_name: dict[int, str] = {}
        self._name_to_label: dict[str, int] = {}
        self.status = "not_trained"   # "not_trained" | "training" | "ready" | "error"
        self.error_message = ""
        self._thread: threading.Thread | None = None

    def train(self, blocking: bool = False):
        """Trigger a retrain. blocking=True waits for completion (used at startup)."""
        if self.status == "training":
            return  # Already in progress

        self.status = "training"
        self.error_message = ""

        if blocking:
            self._do_train()
        else:
            self._thread = threading.Thread(target=self._do_train, daemon=True)
            self._thread.start()

    def _do_train(self):
        try:
            print("[ModelManager] Starting training...")
            # Check if dataset has enough data
            if not os.path.isdir(DATASET_PATH):
                raise RuntimeError(f"Dataset path not found: {DATASET_PATH}")

            enrolled_dirs = [
                d for d in os.listdir(DATASET_PATH)
                if os.path.isdir(os.path.join(DATASET_PATH, d))
            ]
            # Filter to only dirs that actually have images
            valid_dirs = []
            for d in enrolled_dirs:
                img_files = [
                    f for f in os.listdir(os.path.join(DATASET_PATH, d))
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if len(img_files) >= 5:
                    valid_dirs.append(d)

            if len(valid_dirs) < 2:
                raise RuntimeError(
                    f"Need at least 2 subjects with ≥5 images each. Found: {valid_dirs}"
                )

            X_train, y_train, _, _, name_to_label = load_dataset(
                DATASET_PATH, img_size=IMG_SIZE
            )

            if X_train.shape[1] < 2:
                raise RuntimeError("Not enough training images to build PCA model.")

            # PCA
            pca = EigenfacesPCA(k=K_EIGENFACES)
            train_signatures = pca.fit(X_train)  # (p, k)

            # Standardize
            mean_sig = np.mean(train_signatures, axis=0, keepdims=True)
            std_sig = np.std(train_signatures, axis=0, keepdims=True)
            std_sig[std_sig == 0] = 1e-15
            train_signatures_norm = (train_signatures - mean_sig) / std_sig

            # ANN
            num_classes = len(name_to_label)
            ann = MultilayerPerceptron(
                input_dim=train_signatures_norm.shape[1],
                hidden_dim=HIDDEN_DIM,
                output_dim=num_classes,
                lr=0.03,
                reg=0.01,
            )
            ann.train(train_signatures_norm, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

            # Commit atomically
            with self._lock:
                self._pca = pca
                self._ann = ann
                self._mean_sig = mean_sig
                self._std_sig = std_sig
                self._name_to_label = name_to_label
                self._label_to_name = {v: k for k, v in name_to_label.items()}
                self.status = "ready"

            print(f"[ModelManager] Training complete. Classes: {list(name_to_label.keys())}")

        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            print(f"[ModelManager] Training error: {e}")

    def predict(self, frame_bgr) -> dict:
        """
        Accepts a BGR numpy frame, returns prediction dict.
        {
            "name": str,        # predicted name or "Unknown"
            "confidence": float,
            "granted": bool
        }
        """
        import cv2

        with self._lock:
            if self.status != "ready":
                return {"name": "Unknown", "confidence": 0.0, "granted": False,
                        "error": f"Model not ready: {self.status}"}
            pca = self._pca
            ann = self._ann
            mean_sig = self._mean_sig
            std_sig = self._std_sig
            label_to_name = self._label_to_name

        # Preprocess frame
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Use Haar cascade to find face
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))

        if len(faces) == 0:
            return {"name": "No face detected", "confidence": 0.0, "granted": False,
                    "error": "No face detected in frame"}

        # Use largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_roi, IMG_SIZE).astype(np.float32) / 255.0
        face_flat = face_resized.flatten().reshape(-1, 1)  # (mn, 1)

        # PCA project
        face_sig = pca.transform(face_flat)  # (1, k)
        face_sig_norm = (face_sig - mean_sig) / std_sig

        # ANN predict
        probs = ann.forward(face_sig_norm)
        max_prob = float(np.max(probs))
        pred_class = int(np.argmax(probs))

        if max_prob >= CONFIDENCE_THRESHOLD:
            name = label_to_name.get(pred_class, "Unknown")
            granted = True
        else:
            name = "Unknown"
            granted = False

        return {"name": name, "confidence": round(max_prob * 100, 1), "granted": granted}


# Global singleton
manager = ModelManager()

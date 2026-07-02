import os
import cv2
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Dataset Generation & Loading ---
def load_dataset(dataset_path, img_size=(64, 64)):
    """
    Loads face images from dataset_path.
    """
    imposter_names = ['Farhan', 'Ileana']
    
    # Dynamically find all subject folders in the dataset
    all_folders = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    # Enrolled subjects are all folders EXCEPT the designated imposters
    enrolled_names = [d for d in all_folders if d not in imposter_names]
    
    print(f"Dynamically detected enrolled subjects: {enrolled_names}")
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []  # Label will be integer index for enrolled, -1 for imposter
    
    # Map enrolled names to integer labels
    name_to_label = {name: idx for idx, name in enumerate(enrolled_names)}
    
    print("Loading images...")
    
    # Process Enrolled Subjects (60% Train, 40% Test)
    for name in enrolled_names:
        subject_dir = os.path.join(dataset_path, name)
        if not os.path.isdir(subject_dir):
            print(f"Warning: Directory not found for enrolled subject: {name}")
            continue
            
        files = [f for f in os.listdir(subject_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Sort files to ensure deterministic splits
        files.sort()
        
        num_images = len(files)
        num_train = int(0.6 * num_images)
        
        train_files = files[:num_train]
        test_files = files[num_train:]
        
        # Load Train Images
        for f in train_files:
            img_path = os.path.join(subject_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size).astype(np.float32) / 255.0
                train_data.append(img_resized.flatten())
                train_labels.append(name_to_label[name])
                
        # Load Test Images
        for f in test_files:
            img_path = os.path.join(subject_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size).astype(np.float32) / 255.0
                test_data.append(img_resized.flatten())
                test_labels.append(name_to_label[name])
                
    # Process Imposter Subjects (100% Test)
    for name in imposter_names:
        subject_dir = os.path.join(dataset_path, name)
        if not os.path.isdir(subject_dir):
            print(f"Warning: Directory not found for imposter subject: {name}")
            continue
            
        files = [f for f in os.listdir(subject_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            img_path = os.path.join(subject_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size).astype(np.float32) / 255.0
                test_data.append(img_resized.flatten())
                test_labels.append(-1)  # -1 represents "Imposter / Not Enrolled"
                
    X_train = np.array(train_data).T  # Shape: (mn, p)
    y_train = np.array(train_labels)   # Shape: (p,)
    X_test = np.array(test_data).T    # Shape: (mn, p_test)
    y_test = np.array(test_labels)     # Shape: (p_test,)
    
    print(f"Loaded {X_train.shape[1]} training images and {X_test.shape[1]} test images.")
    print(f"Imposter test cases: {np.sum(y_test == -1)} images.")
    return X_train, y_train, X_test, y_test, name_to_label

# --- 2. Custom ANN with Backpropagation from Scratch (L2 Regularized) ---
class MultilayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.03, reg=0.01, beta=0.9):
        self.lr = lr
        self.reg = reg
        self.beta = beta
        
        # He / Xavier Initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Momentum parameters
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
    def relu(self, Z):
        return np.maximum(0, Z)
        
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
        
    def softmax(self, Z):
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
        
    def backward(self, X, y_one_hot):
        N = X.shape[0]
        
        # Output layer gradients
        dZ2 = (self.A2 - y_one_hot) / N
        dW2 = np.dot(self.A1.T, dZ2) + self.reg * self.W2  # L2 regularization
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) + self.reg * self.W1  # L2 regularization
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters with Momentum GD
        self.v_W1 = self.beta * self.v_W1 + (1 - self.beta) * dW1
        self.v_b1 = self.beta * self.v_b1 + (1 - self.beta) * db1
        self.v_W2 = self.beta * self.v_W2 + (1 - self.beta) * dW2
        self.v_b2 = self.beta * self.v_b2 + (1 - self.beta) * db2
        
        self.W1 -= self.lr * self.v_W1
        self.b1 -= self.lr * self.v_b1
        self.W2 -= self.lr * self.v_W2
        self.b2 -= self.lr * self.v_b2
        
    def train(self, X, y, epochs=600, batch_size=32):
        # Convert y to one-hot encoding
        num_classes = self.W2.shape[1]
        y_one_hot = np.eye(num_classes)[y]
        
        N = X.shape[0]
        for epoch in range(epochs):
            # Shuffle at each epoch
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]
            
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

# --- 3. PCA & Eigenfaces Implementation ---
class EigenfacesPCA:
    def __init__(self, k):
        self.k = k
        self.M = None          # Mean face vector
        self.Eigenfaces = None # Eigenfaces matrix (k x mn)
        
    def fit(self, Face_Db):
        """
        Face_Db: Shape (mn, p) where each column is a flattened image vector.
        """
        # Step 2: Mean Calculation
        self.M = np.mean(Face_Db, axis=1, keepdims=True) # (mn, 1)
        
        # Step 3: Mean Zero (alignment)
        A = Face_Db - self.M # (mn, p)
        
        # Step 4: Surrogate Covariance Matrix C = A^T * A
        C = np.dot(A.T, A) # (p, p)
        
        # Step 5: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C) # eigenvalues: (p,), eigenvectors: (p, p)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 6: Select top k eigenvectors
        # Cap k at the maximum possible number of eigenvalues (p)
        actual_k = min(self.k, Face_Db.shape[1] - 1)
        V_selected = eigenvectors[:, :actual_k] # (p, k)
        
        # Step 7: Generating Eigenfaces E = V_selected^T * A^T
        self.Eigenfaces = np.dot(V_selected.T, A.T) # (k, mn)
        
        # Normalize the eigenfaces (each row should be a unit vector)
        row_norms = np.linalg.norm(self.Eigenfaces, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1e-15
        self.Eigenfaces = self.Eigenfaces / row_norms
        
        # Step 8: Generate signatures of training faces
        # Signatures shape: (k, p)
        Signatures = np.dot(self.Eigenfaces, A)
        return Signatures.T # Returns (p, k)
        
    def transform(self, X_test):
        """
        Projects test images onto the eigenfaces.
        X_test: Shape (mn, p_test)
        """
        # Step 2 (testing): Do mean zero
        A_test = X_test - self.M
        # Step 3 (testing): Project test face
        Projected = np.dot(self.Eigenfaces, A_test) # (k, p_test)
        return Projected.T # Returns (p_test, k)

# --- 4. Main Experiment and Evaluation ---
def run_evaluation(dataset_path):
    X_train, y_train, X_test, y_test, name_to_label = load_dataset(dataset_path)
    
    # We will test various values of k
    k_values = [3, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150]
    
    enrolled_accuracies = []
    overall_accuracies = []
    imposter_detection_rates = []
    
    # Softmax confidence threshold for imposter detection
    confidence_threshold = 0.4
    
    print("\nStarting PCA + ANN evaluation across different values of k...")
    print(f"Using Confidence Threshold: {confidence_threshold}")
    print("-" * 80)
    print(f"{'k':<5} | {'Enrolled Acc':<15} | {'Imposter Det Rate':<20} | {'Overall Acc (with imposters)':<28}")
    print("-" * 80)
    
    for k in k_values:
        # 1. PCA Fit and transform
        pca = EigenfacesPCA(k=k)
        train_signatures = pca.fit(X_train) # (p, k)
        test_signatures = pca.transform(X_test) # (p_test, k)
        
        # Standardize features for training stability
        mean_sig = np.mean(train_signatures, axis=0, keepdims=True)
        std_sig = np.std(train_signatures, axis=0, keepdims=True)
        std_sig[std_sig == 0] = 1e-15
        
        train_signatures_norm = (train_signatures - mean_sig) / std_sig
        test_signatures_norm = (test_signatures - mean_sig) / std_sig
        
        # 2. Train Custom ANN (L2 Regularized)
        num_classes = len(name_to_label)
        ann = MultilayerPerceptron(input_dim=train_signatures_norm.shape[1], hidden_dim=128, output_dim=num_classes, lr=0.03, reg=0.01)
        ann.train(train_signatures_norm, y_train, epochs=600, batch_size=32)
        
        # 3. Predictions and Evaluation
        is_enrolled = (y_test != -1)
        X_test_enrolled = test_signatures_norm[is_enrolled]
        y_test_enrolled = y_test[is_enrolled]
        
        # Predict enrolled test images (standard classification)
        probs_enrolled = ann.forward(X_test_enrolled)
        pred_enrolled = np.argmax(probs_enrolled, axis=1)
        enrolled_acc = np.mean(pred_enrolled == y_test_enrolled)
        
        # Predict on ALL test images with imposter thresholding
        probs_all = ann.forward(test_signatures_norm)
        max_probs = np.max(probs_all, axis=1)
        pred_all = np.argmax(probs_all, axis=1)
        
        # Classify as "Not Enrolled" (-1) if confidence is below threshold
        final_preds = np.where(max_probs >= confidence_threshold, pred_all, -1)
        
        # Calculate overall accuracy
        overall_acc = np.mean(final_preds == y_test)
        
        # Calculate imposter detection rate (True Negative Rate)
        imposter_preds = final_preds[~is_enrolled]
        imposter_det_rate = np.mean(imposter_preds == -1)
        
        enrolled_accuracies.append(enrolled_acc)
        overall_accuracies.append(overall_acc)
        imposter_detection_rates.append(imposter_det_rate)
        
        print(f"{k:<5} | {enrolled_acc*100:<13.2f}% | {imposter_det_rate*100:<18.2f}% | {overall_acc*100:<26.2f}%")
        
    print("-" * 80)
    
    # Run a threshold study for k = 80 to show comparative metrics
    print("\nStarting Threshold Study for k = 80...")
    pca_80 = EigenfacesPCA(k=80)
    train_sig_80 = pca_80.fit(X_train)
    test_sig_80 = pca_80.transform(X_test)
    
    mean_sig = np.mean(train_sig_80, axis=0, keepdims=True)
    std_sig = np.std(train_sig_80, axis=0, keepdims=True)
    std_sig[std_sig == 0] = 1e-15
    train_sig_80_norm = (train_sig_80 - mean_sig) / std_sig
    test_sig_80_norm = (test_sig_80 - mean_sig) / std_sig
    
    ann_80 = MultilayerPerceptron(input_dim=train_sig_80_norm.shape[1], hidden_dim=128, output_dim=len(name_to_label), lr=0.03, reg=0.01)
    ann_80.train(train_sig_80_norm, y_train, epochs=600, batch_size=32)
    
    probs_all_80 = ann_80.forward(test_sig_80_norm)
    max_probs_80 = np.max(probs_all_80, axis=1)
    pred_all_80 = np.argmax(probs_all_80, axis=1)
    is_enrolled = (y_test != -1)
    
    thresholds = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    print("-" * 80)
    print(f"{'Threshold':<10} | {'Enrolled Test Acc':<18} | {'Imposter Det Rate':<20} | {'Overall Accuracy':<18}")
    print("-" * 80)
    for t in thresholds:
        final_preds_t = np.where(max_probs_80 >= t, pred_all_80, -1)
        enrolled_correct_t = final_preds_t[is_enrolled] == y_test[is_enrolled]
        enrolled_acc_t = np.mean(enrolled_correct_t)
        
        imposter_correct_t = final_preds_t[~is_enrolled] == -1
        imposter_det_t = np.mean(imposter_correct_t)
        
        overall_acc_t = np.mean(final_preds_t == y_test)
        print(f"{t:<10.2f} | {enrolled_acc_t*100:<16.2f}% | {imposter_det_t*100:<18.2f}% | {overall_acc_t*100:<16.2f}%")
    print("-" * 80)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [a*100 for a in enrolled_accuracies], 'o-', label='Enrolled Accuracy (only enrolled test set)', linewidth=2)
    plt.plot(k_values, [a*100 for a in imposter_detection_rates], 's--', label='Imposter Detection Rate (true negative rate)', linewidth=2)
    plt.plot(k_values, [a*100 for a in overall_accuracies], 'd-.', label=f'Overall Accuracy (threshold={confidence_threshold})', linewidth=2)
    
    plt.title('Face Recognition Accuracy & Imposter Detection vs. Number of Eigenfaces (k)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Eigenfaces (k)', fontsize=12)
    plt.ylabel('Accuracy / Detection Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Save the plot in the current directory and also as an artifact
    plot_name = 'accuracy_vs_k_plot.png'
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved evaluation plot as {plot_name}.")
    
    # Copy plot to the artifacts folder if it exists
    artifacts_dir = 'C:/Users/Administrator/.gemini/antigravity/brain/96c32ab8-a857-42ac-9c8c-48ea119e2abc'
    if os.path.exists(artifacts_dir):
        import shutil
        shutil.copy(plot_name, os.path.join(artifacts_dir, plot_name))
        print("Copied evaluation plot to artifacts folder.")

if __name__ == '__main__':
    dataset_path = 'dataset/dataset/faces'
    run_evaluation(dataset_path)

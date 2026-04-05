# COE4234: COMPUTER VISION AND PATTERN RECOGNITION

## Mid-Term Assignment: Custom CNN for Object Classification

### Student Information
- **Student ID:** 22-47975-2
- **Course Code:** COE4234
- **Course Name:** COMPUTER VISION AND PATTERN RECOGNITION

---

## 1. Project Overview
This project implements a custom Convolutional Neural Network (CNN) using **PyTorch** to classify objects from a subset of the **CIFAR-100** dataset. The model is designed with a 3-block architecture to distinguish between 10 specific classes consisting of animals and vehicles.

### Selected Classes
- **Animals:** `bear`, `elephant`, `lion`, `tiger`, `wolf`
- **Vehicles:** `bicycle`, `bus`, `motorcycle`, `pickup_truck`, `train`

---

## 2. Methodology & Architecture
The implementation follows a modular 3-block CNN approach:
1.  **Block 1:** 32 filters, 3x3 kernel, ReLU activation.
2.  **Block 2:** 64 filters, 3x3 kernel, ReLU activation.
3.  **Block 3:** 128 filters, 3x3 kernel, ReLU activation.
4.  **Regularization:** Integrated **Batch Normalization** and **Dropout** (p=0.3) to enhance generalization.
5.  **Classification Head:** Fully connected layers reducing 128 channels to 10 classes.

### Hyperparameters
- **Optimizer:** Adam
- **Scheduler:** CosineAnnealingLR
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 64

---

## 3. Results & Evaluation
The model achieved a **Test Accuracy of 78.50%** after 30 epochs of training.

### Key Performance Metrics
- **Macro F1-Score:** ~77.8%
- **Best Performing Class:** `motorcycle` (88.46%)
- **Worst Performing Class:** `bus` (60.00%)

### Visualizations Included
- Training/Validation Loss and Accuracy Curves.
- **Ablation Study:** Comparison of model performance with and without regularization (BN & Dropout).
- **Confusion Matrix:** Raw and Normalized performance per class.
- **Per-Class F1 Chart:** Visual bar representation of classification strength.

---

## 4. Deliverables
As per the assignment requirements, the following files are provided in this repository:
1.  `CNN_22-47975-2.ipynb`: Complete implementation with code, outputs, and documentation.
2.  `CNN_22-47975-2.pth`: Final trained model weights for verification.

---

## 5. Setup & Usage
1.  **Environment:** Requires Python 3.8+, PyTorch, Torchvision, Scikit-learn, and Matplotlib.
2.  **Execution:** Run all cells in `CNN_22-47975-2.ipynb`. The dataset will be automatically downloaded to the `data_subset/` directory.

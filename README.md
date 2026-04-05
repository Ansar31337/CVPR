# COE4234: COMPUTER VISION AND PATTERN RECOGNITION
## Mid-Term Assignment: Custom CNN for Object Classification

### Student Information
- **Student ID:** 22-47975-2
- **Course Code:** COE4234
- **Course Name:** COMPUTER VISION AND PATTERN RECOGNITION

---

## 1. Project Overview
This repository contains the complete implementation of a custom Convolutional Neural Network (CNN) using **PyTorch** to classify objects from the **CIFAR-100** dataset. As per the assignment requirements, the model focuses on a subset of 10 classes, evaluating the impact of regularization and advanced optimization techniques.

### Selected Classes (Subset of CIFAR-100)
- **Animals:** `bear`, `elephant`, `lion`, `tiger`, `wolf`
- **Vehicles:** `bicycle`, `bus`, `motorcycle`, `pickup_truck`, `train`

---

## 2. Implementation Methodology

### Data Preprocessing
- **Normalization:** Calculated and applied mean/std normalization specific to the 10-class subset.
- **Data Splitting:** Implemented a mandatory **10% Validation Split** from the training set.
- **Batch Size:** 64.

### Model Architecture
The model follows a modular 3-block CNN design:
1.  **Block 1:** Conv2D (3 -> 32 filters, 3x3), BatchNorm, ReLU, MaxPool2D, Dropout (0.3).
2.  **Block 2:** Conv2D (32 -> 64 filters, 3x3), BatchNorm, ReLU, MaxPool2D, Dropout (0.3).
3.  **Block 3:** Conv2D (64 -> 128 filters, 3x3), BatchNorm, ReLU, MaxPool2D, Dropout (0.3).
4.  **Classification Head:** Global Average Pooling and Dense layers to map 128 channels to 10 classes.

### Optimization & Training
- **Optimizer:** Adam.
- **Scheduler:** CosineAnnealingLR for smooth learning rate decay.
- **Loss Function:** CrossEntropyLoss.
- **Epochs:** 30.

---

## 3. Results & Evaluation
The model achieved a **Test Accuracy of 78.50%**.

### Key Performance Summary
- **Macro F1-Score:** 77.85%
- **Best Performing Class:** `motorcycle` (88.46% F1-score)
- **Worst Performing Class:** `bus` (60.00% F1-score)

### Visualizations Provided
- **Training/Validation Curves:** Monitoring loss and accuracy over epochs.
- **Ablation Study:** Comparative curves showing performance with and without Regularization (BN/Dropout).
- **Confusion Matrix:** Detailed visualization of classification confusion between similar classes (e.g., Tiger vs. Wolf).
- **Per-Class F1 Chart:** Bar chart representing model confidence across all 10 categories.

---

## 4. Deliverables
As specified in Section 5 of `CVPR_Mid.pdf`:
1.  **`CNN_22-47975-2.ipynb`**: The complete Jupyter notebook with code, execution outputs, and visualizations.
2.  **`CNN_22-47975-2.pth`**: The saved state dictionary of the final trained model.

---

## 5. How to Run
1.  Ensure you have `torch`, `torchvision`, `scikit-learn`, `matplotlib`, and `seaborn` installed.
2.  Open `CNN_22-47975-2.ipynb` in Jupyter or Google Colab.
3.  Run all cells; the CIFAR-100 dataset will be automatically downloaded to the `data_subset/` folder.

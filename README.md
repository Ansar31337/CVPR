# COE4234: COMPUTER VISION AND PATTERN RECOGNITION
## Mid-Term Assignment: Deep Learning for Object Classification

### Course Information
*   **Course Code:** COE4234
*   **Course Name:** COMPUTER VISION AND PATTERN RECOGNITION
*   **Department:** Computer Science & Engineering

---

## 1. Executive Summary
This project presents a specialized Convolutional Neural Network (CNN) architecture developed using **PyTorch** to solve a 10-class object recognition task on a subset of the **CIFAR-100** dataset. The model integrates modern deep learning components, including **Batch Normalization**, **Spatial Dropout**, and **Global Average Pooling**, achieving a robust test accuracy of **78.50%**.

---

## 2. Dataset Specification
The implementation utilizes a curated subset of the CIFAR-100 dataset, focusing on two distinct super-classes:

| Category | Classes |
| :--- | :--- |
| **Animals** | `bear`, `elephant`, `lion`, `tiger`, `wolf` |
| **Vehicles** | `bicycle`, `bus`, `motorcycle`, `pickup_truck`, `train` |

### Data Pipeline
*   **Preprocessing:** 32x32 resolution resizing, per-channel normalization (subset-specific mean/std).
*   **Data Splitting:** A fixed-seed **10% Validation Split** (500 samples) was extracted from the 5,000-sample training subset to ensure reproducible hyperparameter tuning.

---

## 3. Technical Architecture
The model features a modular **3-Block Convolutional** design with an increasing receptive field and channel depth:

### Feature Extraction (Encoder)
1.  **Block 1:** Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
2.  **Block 2:** Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
3.  **Block 3:** Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.3)

### Classification Head (Decoder)
*   **Global Average Pooling:** Used to reduce spatial dimensions while preserving feature map activation intensity.
*   **Dense Layers:** Fully connected layers with ReLU activation mapping to the final 10-class output.

---

## 4. Training Strategy & Optimization
*   **Optimizer:** `Adam` (Adaptive Moment Estimation) for efficient gradient descent.
*   **Learning Rate Policy:** `CosineAnnealingLR` scheduler to simulate warm restarts and prevent local minima entrapment.
*   **Loss Criterion:** `CrossEntropyLoss` (Log-Softmax integrated) for multi-class optimization.
*   **Batch Size:** 64 samples per iteration.

---

## 5. Performance Evaluation
The model was rigorously evaluated across several metrics to ensure generalization:

### Quantitative Results
*   **Overall Test Accuracy:** 78.50%
*   **Macro F1-Score:** 77.85%
*   **Class-Specific Insights:**
    *   **Highest Performance:** `motorcycle` (**88.46%** F1-score) — High geometric distinctness.
    *   **Lowest Performance:** `bus` (**60.00%** F1-score) — Significant confusion with `train` due to similar structural aspect ratios.

### Included Visualizations
*   Learning Curves (Training vs. Validation Loss/Accuracy).
*   **Ablation Study:** Validation of the positive impact of BatchNorm and Dropout on model convergence.
*   **Confusion Matrix:** Heatmap analysis of inter-class misclassifications.
*   **F1-Score Distribution:** Per-class performance bar chart.

---

## 6. Repository Deliverables
In accordance with the assignment requirements, this repository contains:
*   `CNN_22-47975-2.ipynb`: Complete source code, experimental outputs, and visualization suite.
*   `CNN_22-47975-2.pth`: Serialized model state dictionary (weights) for deployment/verification.

---

## 7. Installation & Reproduction
1.  Clone the repository and ensure the following dependencies are installed: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`.
2.  Execute the Jupyter Notebook `CNN_22-47975-2.ipynb`.
3.  The dataset will be automatically downloaded and processed within the local environment.

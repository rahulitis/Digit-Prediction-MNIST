# MNIST Handwritten Digit Classification

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Performance Analysis](#performance-analysis)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## Overview

This project implements a comprehensive machine learning pipeline for classifying handwritten digits from the MNIST dataset. The implementation progresses from binary classification using SGD to advanced multi-class classification with optimized KNN, achieving 98% accuracy through strategic hyperparameter tuning and data augmentation.

**Project Highlights**:
- **Multi-stage Analysis**: From binary (SGD) to multi-class (KNN) classification
- **Performance Optimization**: Systematic hyperparameter tuning and data augmentation
- **Robust Evaluation**: Cross-validation methodology ensuring unbiased results
- **Visual Analysis**: Comprehensive confusion matrix analysis revealing model insights

### Key Features
- Multi-classifier comparison and evaluation
- Custom data augmentation pipeline
- Hyperparameter optimization
- Comprehensive performance visualization
- Detailed error analysis and model interpretation

## Dataset

The MNIST dataset is a benchmark dataset in machine learning, consisting of:

| Property | Value |
|----------|-------|
| **Total Samples** | 70,000 |
| **Training Samples** | 60,000 |
| **Test Samples** | 10,000 |
| **Image Dimensions** | 28×28 pixels |
| **Feature Vector Size** | 784 (flattened) |
| **Classes** | 10 (digits 0-9) |
| **Color Space** | Grayscale (0-255) |

The dataset is automatically fetched using `sklearn.datasets.fetch_openml('mnist_784')`, eliminating the need for manual downloads.

## Installation

### Requirements

```bash
pip install scikit-learn matplotlib numpy jupyter
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.0.0 | Machine learning algorithms and dataset |
| `matplotlib` | ≥3.5.0 | Data visualization |
| `numpy` | ≥1.21.0 | Numerical computations |
| `jupyter` | ≥1.0.0 | Interactive notebook environment |

## Model Architecture

### 1. Stochastic Gradient Descent (SGD)
```python
SGDClassifier(random_state=42)
```
- **Advantages**: Fast training, memory efficient
- **Use Case**: Large-scale learning, online learning scenarios

### 2. Random Forest Classifier
```python
RandomForestClassifier(random_state=42)
```
- **Advantages**: Robust to overfitting, handles non-linear patterns
- **Use Case**: Ensemble learning, feature importance analysis

### 3. K-Nearest Neighbors (KNN)
```python
KNeighborsClassifier(n_neighbors=3, weights='distance')
```
- **Optimized Parameters**:
  - `n_neighbors`: 3 (determined through hyperparameter tuning)
  - `weights`: 'distance' (closer neighbors have higher influence)

## Data Augmentation

### Augmentation Strategy

The dataset is expanded through spatial transformations to improve model robustness:

```python
def DataAugmentor(data):
    """
    Augments a single 28x28 image by creating 4 shifted versions
    Returns: top, bottom, left, right shifted images
    """
    data = data.reshape(28, 28)
    
    # Create padding arrays
    row_zeros = np.zeros(28)
    column_zeros = np.zeros((28, 1))
    
    # Generate shifted versions
    leftTrimSub = data[:, 1:]     # Remove leftmost column
    rightTrimSub = data[:, :-1]   # Remove rightmost column
    topTrimSub = data[1:, :]      # Remove top row
    bottomTrimSub = data[:-1, :]  # Remove bottom row
    
    # Add padding to create shifts
    top_shift = np.vstack([row_zeros, bottomTrimSub])
    bottom_shift = np.vstack([topTrimSub, row_zeros])
    left_shift = np.hstack([column_zeros, rightTrimSub])
    right_shift = np.hstack([leftTrimSub, column_zeros])
    
    return (top_shift.flatten(), bottom_shift.flatten(), 
            left_shift.flatten(), right_shift.flatten())
```
<img width="822" height="199" alt="image" src="https://github.com/user-attachments/assets/75a97dcb-e0c5-40c4-84e9-62a88af108ab" />

### Augmentation Results
- **Original Dataset**: 70,000 samples  
- **Augmented Dataset**: 350,000 samples (5× increase)
- **Transformation Method**: Four directional pixel shifts (top, bottom, left, right)
- **Accuracy Improvement**: KNN gained 1% accuracy boost (97% → 98%)
- **Implementation**: Custom `DataAugmentor` function creating spatially shifted versions
<img width="502" height="432" alt="image" src="https://github.com/user-attachments/assets/4b244f42-912e-4dbb-b59a-469a6104e704" />

## Hyperparameter Tuning

### K-Nearest Neighbors Optimization

The KNN classifier underwent systematic hyperparameter tuning, resulting in significant performance improvements:

| Parameter | Options Tested | Optimal Value | Performance Impact |
|-----------|----------------|---------------|-------------------|
| `n_neighbors` | [1, 3, 5, 7, 9] | 3 | Balanced bias-variance trade-off |
| `weights` | ['uniform', 'distance'] | 'distance' | Better handling of varying distances |

**Performance Progression**:
- Base KNN: ~94% accuracy
- Tuned KNN: ~97% accuracy (+3% improvement)
- Tuned KNN + Augmentation: ~98% accuracy (+1% additional improvement)

**Note**: Hyperparameter tuning was the most computationally intensive phase, requiring extensive cross-validation across parameter combinations.

## Results

### Model Performance Comparison

| Model | Task | Accuracy (Original) | Accuracy (Augmented) | Key Features |
|-------|------|-------------------|---------------------|--------------|
| SGD Classifier | Binary (Digit 5) | ~90% | - | Fast training, memory efficient |
| Random Forest | Multi-class | ~96% (scaled data) | - | Robust ensemble method |
| KNN (Original) | Multi-class | ~94% | - | Instance-based learning |
| KNN (Tuned) | Multi-class | ~97% | **98%** | Optimized parameters + augmentation |

### Performance Highlights
- **Best Overall Performance**: KNN with hyperparameter tuning and data augmentation (98% accuracy)
- **Data Scaling Impact**: Random Forest improved significantly with feature scaling
- **Augmentation Benefit**: KNN gained 1% accuracy improvement with augmented dataset
- **Cross-Validation**: All models evaluated using cross-validation for robust performance estimates

### Evaluation Metrics
- **Binary Classification (SGD)**: Precision-recall curves and ROC analysis for digit 5 detection
- **Multi-class Classification**: Confusion matrices showing per-class performance
- **Cross-Validation Scores**: Used `cross_val_predict` for unbiased model evaluation
- **Confusion Matrix Analysis**: Detailed error patterns and misclassification insights

## Usage

### Quick Start

1. **Clone and Setup**:
```bash
git clone [repository-url]
cd mnist-classification
pip install -r requirements.txt
```

2. **Run the Notebook**:
```bash
jupyter notebook MNIST.ipynb
```

3. **Execute Cells Sequentially**:
   - Data loading and exploration
   - Visualization of sample digits
   - Data augmentation (optional)
   - Model training and comparison
   - Performance evaluation and visualization

### Example Output

```python
# Sample digit visualization
first_digit = mnist.data[0]
first_label = mnist.target[0]
print(f"Label: {first_label}")
Plotter(first_digit)  # Displays 28x28 digit image
```

## File Structure

```
mnist-classification/
│
├── MNIST.ipynb              # Main notebook with implementation
├── README.md                # This documentation
├── requirements.txt         # Python dependencies
├── results/                 # Output directory
│   ├── confusion_matrices/  # Confusion matrix plots
│   ├── precision_recall/    # PR curve plots
│   └── roc_curves/         # ROC curve plots
└── models/                 # Saved model files (optional)
```

## Performance Analysis

## Performance Analysis

### Confusion Matrix Analysis

The project includes two key confusion matrix visualizations:

1. **Initial SGD Binary Classifier** (Digit 5 Detection):
   - Shows precision-recall trade-offs for binary classification
   - Demonstrates SGD performance on single-digit detection task
   - Accuracy: ~90%

2. **Final KNN Multi-class Results** (After Hyperparameter Tuning + Augmentation):
   - Near-perfect diagonal pattern indicating excellent classification
   - Minimal off-diagonal errors showing rare misclassifications
   - Accuracy: 98%
   - Strongest performance on digits 0, 2, 6, 8 (near 99% accuracy)
   - Most challenging digits: 3, 5, 8, 9 (slight confusion patterns visible)

### Cross-Validation Methodology

All models were evaluated using scikit-learn's `cross_val_predict` to ensure:
- **Unbiased Performance Estimates**: Each sample predicted using a model not trained on it
- **Robust Model Selection**: Consistent performance across different data splits
- **Decision Analysis**: Understanding model behavior on edge cases

### Key Insights from Confusion Matrices
- **Digit Pairs with Confusion**: 
  - 3 ↔ 5: Structural similarities in handwriting
  - 8 ↔ 9: Similar curved patterns
  - 4 ↔ 9: Overlapping geometric features
- **Most Accurate Digits**: 0, 1, 6 show near-perfect classification
- **Data Augmentation Impact**: Improved generalization reduced misclassification rates

## Future Improvements

### Immediate Enhancements
- [ ] **Advanced Preprocessing**: Pixel normalization and contrast enhancement
- [ ] **Cross-Validation**: K-fold validation for robust performance estimates
- [ ] **Model Serialization**: Save/load trained models using `joblib`
- [ ] **Batch Prediction**: Efficient inference pipeline

### Advanced Features
- [ ] **Deep Learning Integration**: CNN comparison using TensorFlow/PyTorch
- [ ] **Ensemble Methods**: Voting and stacking classifiers
- [ ] **Real-time Prediction**: Web interface for digit recognition
- [ ] **Performance Profiling**: Memory and computational efficiency analysis

### Research Directions
- [ ] **Novel Augmentation Techniques**: Rotation, scaling, elastic deformations
- [ ] **Feature Engineering**: HOG, SIFT, or learned features
- [ ] **Transfer Learning**: Pre-trained feature extractors
- [ ] **Adversarial Robustness**: Evaluation against adversarial examples

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST dataset creators and maintainers
- Scikit-learn community for robust ML tools
- Open-source contributors to visualization libraries

---

**Note**: This project serves as an educational demonstration of classical machine learning techniques on a benchmark dataset. For production applications, consider more advanced deep learning approaches.

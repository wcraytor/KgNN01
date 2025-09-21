## PRELIMINARY NOTES AND BACKGROUND

- This is a Kaggle Jupyter and Python project, which is part of a network of open source dealing with "Valuation Engineering."
- For contextual background; ["Residual Constraint Approach"](https://doi.org/10.5281/zenodo.14787917)
- Craytor, William B. (2025). Residual Constraint Approach (RCA). Zenodo. DOI:  [10.5281](https://doi.org/10.5281/zenodo.14787917[](https://doi.org/10.1000/182))(https://doi.org/10.5281/zenodo.14787917)

## Room Classification with PyTorch

A deep learning-based room type classification system that identifies 10 different interior room types from photographs using transfer learning and computer vision techniques.

## Overview

This project implements a convolutional neural network (CNN) using PyTorch to classify interior photographs into one of 10 room categories. The system uses transfer learning with ResNet50 architecture and includes comprehensive data management, training pipeline, and evaluation metrics.

## Features

* **10 Room Types**: Bathroom, bedroom, dining, gaming, kitchen, laundry, living, office, terrace, yard
* **Transfer Learning**: ResNet50 backbone with custom classifier head
* **Flexible Data Sampling**: Use 5%-100% of your dataset for different experiment scales
* **Hardware Optimization**:
  * Apple Silicon (MPS) support for M1/M2/M3 Macs
  * NVIDIA GPU (CUDA) acceleration
  * CPU fallback for systems without GPU
* **Memory Efficient**: Optimized for 64GB RAM systems with configurable batch sizes
* **Comprehensive Evaluation**: Per-room accuracy analysis with detailed reporting
* **Advanced Training**: AdamW optimizer, cosine annealing, label smoothing, gradient clipping

## Dataset Structure

Organize your photos in the following folder structure:

```
images/
├── bathroom/
│   ├── photo001.jpg
│   ├── photo002.jpg
│   └── ...
├── bedroom/
├── dining/
├── gaming/
├── kitchen/
├── laundry/
├── living/
├── office/
├── terrace/
└── yard/
```

## Installation

### Requirements

* Python 3.8+
* PyTorch 1.12+
* CUDA-compatible GPU (optional, recommended for faster training)

### Dependencies

```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn pandas pillow
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from room_classifier import train_room_classifier

# Quick test with 25% of your photos
model, history, dataset_manager = train_room_classifier(
    base_path='images/',
    sample_percentage=25,
    epochs=10,
    batch_size=16
)
```

### Full Training

```python
# Train on complete dataset for maximum accuracy
model, history, dataset_manager = train_room_classifier(
    base_path='images/',
    sample_percentage=100,
    epochs=20,
    batch_size=16
)
```

## Configuration Options


| Parameter           | Description                 | Default     | Recommended Values                               |
| ------------------- | --------------------------- | ----------- | ------------------------------------------------ |
| `base_path`         | Path to images folder       | `'images/'` | Your photo directory                             |
| `sample_percentage` | Percentage of photos to use | `100`       | `5`(testing),`25`(development),`100`(production) |
| `epochs`            | Number of training epochs   | `20`        | `5-10`(testing),`20-30`(production)              |
| `batch_size`        | Batch size for training     | `16`        | `8-16`(64GB RAM),`32+`(128GB+ RAM)               |
| `dataset_size`      | Dataset scale               | `'full'`    | `'tiny'`,`'small'`,`'medium'`,`'large'`,`'full'` |

## Performance

### Expected Accuracy

* **Training from scratch**: 75-85%
* **With transfer learning**: 85-95%
* **Dataset size impact**: \~2-5% improvement using full vs 25% of photos

### Training Times (Apple M2 Ultra, 64GB RAM)

* **25% dataset, 10 epochs**: \~90 minutes
* **100% dataset, 20 epochs**: \~3 hours
* **Quick test (5%, 3 epochs)**: \~15 minutes

### Memory Requirements

* **Minimum**: 8GB RAM (CPU training)
* **Recommended**: 64GB RAM (GPU training)
* **Batch size**: 16 for 64GB systems, 8 for 32GB systems

## Output Files

The training process generates several analysis files:

* `room_classification_results.txt` - Overall performance and confusion matrix
* `TestResultsByRoomType.log` - Per-room accuracy breakdown
* `EpochStatistics.log` - Training progress data
* `room_training_history.png` - Loss and accuracy curves
* `room_confusion_matrix.png` - Visual confusion matrix

## Advanced Features

### Data Sampling

Control dataset size for faster experimentation:

```python
# Use only 10% of photos for rapid prototyping
dataset = RoomPhotoDataset('images/', sample_percentage=10)
```

### Custom Model Architecture

```python
# Modify model parameters
model = OptimizedRoomNet(
    num_classes=10,
    dropout_rate=0.3  # Adjust regularization
)
```

### Hardware Optimization

The system automatically detects and uses the best available hardware:

* Apple Silicon (MPS) for M1/M2/M3 Macs
* NVIDIA GPU (CUDA) for dedicated graphics cards
* CPU fallback for compatibility

## Testing

The project includes comprehensive test functions for development:

```python
# Test individual components
dataset = test_dataset_loading('images/', sample_percentage=1)
model = test_model_creation()
device = test_device_detection()

# Run complete test suite
success = run_comprehensive_test('images/', sample_percentage=0.5)
```

## Troubleshooting

### Common Issues

**Memory Errors**

```python
# Reduce batch size
train_room_classifier(batch_size=8)  # Instead of 16
```

**SSL Certificate Errors**

* The code includes automatic SSL bypass for PyTorch downloads
* If issues persist, training will fall back to non-pre-trained models

**Missing Room Folders**

* System continues with available rooms and shows warnings
* Ensure all 10 room type folders exist in your images directory

**Slow Training**

* Use smaller `sample_percentage` during development
* Enable GPU acceleration if available
* Reduce image resolution in transforms if needed

### Performance Optimization

**For Development Speed**:

```python
# Fast iteration setup
train_room_classifier(
    sample_percentage=5,    # Very small dataset
    epochs=3,              # Few epochs
    batch_size=8           # Conservative memory usage
)
```

**For Maximum Accuracy**:

```python
# Production setup
train_room_classifier(
    sample_percentage=100,  # Full dataset
    epochs=25,             # Extended training
    batch_size=16          # Optimized batch size
)
```

## Technical Details

### Model Architecture

* **Backbone**: ResNet50 convolutional neural network
* **Transfer Learning**: Pre-trained on ImageNet (optional)
* **Classifier**: Custom fully connected layers with dropout
* **Optimization**: AdamW optimizer with cosine annealing schedule

### Data Pipeline

* **Image Preprocessing**: Resize to 512×512, normalization
* **Data Augmentation**: Random crops, flips, rotation, color jittering
* **Memory Management**: On-demand loading, configurable batch sizes
* **Split Strategy**: Stratified 70/15/15 train/validation/test split

### Evaluation Metrics

* Overall classification accuracy
* Per-room type accuracy analysis
* Confusion matrix with misclassification patterns
* Precision, recall, and F1-score per class

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## Acknowledgments

* PyTorch team for the deep learning framework
* ResNet architecture from "Deep Residual Learning for Image Recognition"
* Transfer learning techniques from torchvision model zoo

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{room_classification_2024,
  title={Room Classification with PyTorch},
  author={Your Name},
  year={2024},
  url={https://github.com/wcraytor/KgNN01}
}
```

## Contact

For questions or issues, please open a GitHub issue or email [bcraytor@proton.me].

---

**Project Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: December 2024

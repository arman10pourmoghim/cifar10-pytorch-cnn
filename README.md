# **CIFAR-10 PyTorch Training Pipeline**

A fully modular PyTorch training pipeline for the CIFAR-10 image classification dataset.
This project implements a clean end-to-end workflow including:

* Custom CNN architecture
* Data loading and strong data augmentation
* Training/validation loops with accurate device management
* Learning-rate schedulers (CosineAnnealingLR / StepLR)
* Early stopping
* Reproducibility features
* Evaluation metrics, confusion matrix, and misclassification analysis

This repository is structured to reflect real-world deep learning engineering practices and to serve as a foundation for further experimentation or extension into deeper models (e.g., ResNet, EfficientNet).

---

## **Features**

### **Model**

* Custom convolutional neural network (CNN)
* Modular `nn.Module` definition for easy extension
* ReLU activation, MaxPooling, Dropout
* Classification head with configurable hidden dimensions

### **Training Pipeline**

* Separate `train_one_epoch` and `evaluate` functions
* GPU/CPU automatic device selection
* Adam or SGD optimizers
* Learning-rate schedulers:

  * `CosineAnnealingLR`
  * `StepLR` (optional)
* Gradient handling and optimizer reset
* Best-model snapshot saving

### **Regularization**

* Weight decay (L2 regularization)
* Dropout in classifier head
* Strong data augmentation:

  * RandomCrop
  * RandomHorizontalFlip
  * RandomRotation
  * ColorJitter
  * RandomErasing

### **Early Stopping**

* Monitors validation loss
* Stops training when improvement stalls
* Restores best model weights automatically

### **Evaluation**

* Accuracy and loss tracking
* Confusion matrix (via scikit-learn)
* Visualization of misclassified samples
* Support for restoring checkpoints

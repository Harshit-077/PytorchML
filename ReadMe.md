# 🌾 Rice Type Classification using PyTorch

A Binary Classification Deep Learning project built with **PyTorch** to classify rice types based on morphological features.

This project trains a simple neural network to classify rice grains into two categories using a Kaggle dataset.

---

## 📌 Project Overview

- **Framework:** PyTorch  
- **Dataset:** Rice Type Classification (Kaggle)  
- **Problem Type:** Binary Classification  
- **Model Type:** Fully Connected Neural Network  
- **Final Test Accuracy:** **98.53%**

---

## 📂 Dataset

The dataset is downloaded using `opendatasets` directly from Kaggle.
`https://www.kaggle.com/datasets/mssmartypants/rice-type-classification`

Features include:

- Area  
- MajorAxisLength  
- MinorAxisLength  
- Eccentricity  
- ConvexArea  
- EquivDiameter  
- Extent  
- Perimeter  
- Roundness  
- AspectRation  

Target:
- `Class` → 0 or 1 (Rice Type)

Dataset Size:
- 18,185 samples  
- 10 input features  
- Binary output  

---

## ⚙️ Installation & Setup

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install opendatasets
pip install torchsummary
pip install scikit-learn
pip install matplotlib
pip install pandas numpy
```

### 2️⃣ Install Dependencies
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")
```

## 🧠 Model Architecture
Simple Feedforward Neural Network:  
Input Layer → 10 Features  
Hidden Layer → 10 Neurons  
Output Layer → 1 Neuron  

Activation → Sigmoid  
Linear(10 → 10)  
Linear(10 → 1)  
Sigmoid  

Total Trainable Parameters: 121  
Loss Function:
Binary Cross Entropy Loss (BCELoss)  
Optimizer:
Adam (learning rate = 1e-3)
## 📊 Training Details
Epochs: 10  
Batch Size: 32  
Train/Validation/Test Split:  
70% Train  
15% Validation  
15% Test  
## 📈 Final Results
Metric	Value  
Training Accuracy	98.66%  
Validation Accuracy	98.75%  
Test Accuracy	98.53%  
## 📉 Training Curves
The model shows:  
Smooth decrease in training & validation loss  
Stable high validation accuracy  
No major overfitting observed  
## 🔮 Sample Prediction
The model can predict custom inputs after normalization:  
my_prediction = model(input_tensor)  
print(my_prediction.item())  
Example Output:
1.0
## 🚀 Key Learnings
Building custom PyTorch Dataset classes  
Using GPU / MPS device selection  
Implementing train-validation-test split  
Tracking loss and accuracy manually  
Plotting training curves  
Performing inference on new data  
## 📁 Future Improvements
Add more hidden layers  
Try ReLU instead of linear-only hidden layer  
Use BCEWithLogitsLoss (remove sigmoid from model)  
Add early stopping  
Add confusion matrix visualization  
Convert into a Streamlit web app  
## 🧑‍💻 Author
Harshit Sharma  
Computer Science Student

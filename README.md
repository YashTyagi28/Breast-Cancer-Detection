# Breast Cancer Detection System

This project is a **Breast Cancer Detection System** designed to predict whether breast cancer cells are malignant (cancerous) or benign (non-cancerous). It uses deep learning and interactive visualization for reliable predictions, aimed at assisting healthcare professionals in early cancer diagnosis.

---

## 📋 **Overview**

- **Frameworks/Tools**: PyTorch, Streamlit, Plotly, Pandas, scikit-learn.
- **Core Features**:
  - Predicts if cell clusters are malignant or benign based on 30 input features.
  - Visualizes cell measurements using an interactive radar chart.
  - Provides real-time predictions with probabilities.

---

## 🚀 **How It Works**

1. **Input Features**:
   - Users can input 30 features derived from cell nuclei measurements, categorized into:
     - Mean
     - Standard Error (SE)
     - Worst Value
   - Features include radius, texture, perimeter, area, compactness, concavity, and more.

2. **Prediction**:
   - A PyTorch-based neural network classifies input data as **Malignant** or **Benign**.
   - The classification is probabilistic, displaying both confidence scores and diagnostic results.

3. **Visualization**:
   - A radar chart shows scaled values of features across Mean, SE, and Worst Value categories, providing a comprehensive view of the cell measurements.

---

## 📁 **File Descriptions**

### **App Structure**
- `app/main.py`:  
  Streamlit application handling user input, radar chart visualization, and predictions. Key features:
  - Sidebar with sliders for all 30 features.
  - Integration with the PyTorch-trained model for predictions.
  - Generates a radar chart comparing Mean, SE, and Worst Value categories.

- `data/data.csv`:  
  Dataset containing breast cancer cell nuclei measurements with 30 features and a diagnosis column (`B` for benign, `M` for malignant).

- `model/saved_model.pt`:  
  PyTorch model file with trained weights for predicting malignancy.

- `model/scaler.pkl`:  
  Pickle file storing the scaler object for feature normalization.

### **Model Creation**
- `model/main.py`:  
  Code for data preprocessing, model architecture, training, and saving:
  - **Architecture**: 30 input features → 8 hidden units → 1 output.
  - Uses **BCEWithLogitsLoss** for binary classification.
  - Trained with Adam optimizer over 100 epochs.
  - Splits data into train-test (80-20%) and evaluates accuracy after each epoch.

### **Requirements**
- `requirements.txt`:  
  Lists all dependencies for running the project:
  - pandas
  - streamlit
  - numpy
  - torch
  - scikit-learn
  - plotly

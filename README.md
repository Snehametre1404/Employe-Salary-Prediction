# 🧑‍💼 Employee Salary Prediction Web App

This project is a **machine learning-based web application** built with **Streamlit** to predict whether an individual earns more than $50K per year using the **Adult Income Dataset**. It uses a trained **K-Nearest Neighbors (KNN)** classifier and includes dynamic, label-aware input fields for intuitive user experience.

---

## 📊 Dataset

- **Source**: UCI Adult Income Dataset
- **Target Variable**: `income` (<=50K or >50K)
- **Features Used**:  
  `age`, `workclass`, `fnlwgt`, `education`, `educational-num`,  
  `marital-status`, `occupation`, `relationship`, `race`, `gender`,  
  `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`

---

## 🚀 Features

- Clean and encode categorical data using `LabelEncoder`
- Trained on KNN algorithm using `scikit-learn`
- Saves model as `knn_model.pkl` and encoders as `label_encoders.pkl`
- Intuitive web interface using `Streamlit`
- Dynamic dropdowns show **readable categories** (e.g., "Male", "Private", etc.)
- Predicts whether a person earns `<=50K` or `>50K` based on user input

---

## 🧠 Model Accuracy

- Achieved accuracy: **~82%**
- Algorithm: **K-Nearest Neighbors (n_neighbors=5)**

---

## 🛠 How to Run Locally

### 1. Install Dependencies


pip install streamlit pandas scikit-learn joblib


### 2. Train the Model
Make sure adult.csv is placed in the project directory. Then run:

python train_model.py

This will generate:

- knn_model.pkl – trained KNN model
- label_encoders.pkl – label encoders for dropdowns

 ### 3. Launch the Streamlit App
 
streamlit run employee_salary_app.py

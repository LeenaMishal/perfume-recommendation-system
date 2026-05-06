# Perfume Recommendation System

A machine learning-based perfume recommendation system that predicts suitable perfume categories based on user preferences and perfume characteristics.

The project demonstrates the complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment using Streamlit.

---

## Project Overview

The main goal of this project is to build an intelligent recommendation system that helps users discover perfumes that match their preferences.

The system predicts perfume recommendations using several input features related to:
- Gender
- Age
- Occasion
- Weather
- Perfume strength
- Scent preferences

The project also includes a graphical user interface (GUI) built using Streamlit to allow users to interact with the model easily.

---

## Dataset Description

The dataset contains perfume-related information used to train machine learning classification models.

The dataset includes:
- Numerical features
- Categorical features
- Target perfume categories

### Example Features
- Gender
- Age Group
- Occasion
- Weather
- Perfume Strength
- Preferred Notes

### Target Variable
- Recommended perfume category or perfume type

---

## Data Preprocessing

Several preprocessing techniques were applied before training the machine learning models.

### Data Cleaning
- Checked for missing values
- Removed duplicated records
- Verified dataset consistency

### Encoding
Categorical variables were converted into numerical values using encoders to make the dataset compatible with machine learning algorithms.

Encoded features include:
- Gender
- Occasion
- Weather
- Perfume categories

### Feature Scaling
Numerical features were normalized using StandardScaler to improve model performance and maintain balanced feature ranges.

---

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to better understand the dataset and identify important patterns.

### Analysis Included
- Distribution of perfume categories
- User preference analysis
- Correlation between features
- Data visualization using charts and plots

### Visualizations
The project includes:
- Bar charts
- Histograms
- Correlation analysis
- Category distribution plots

EDA helped identify trends and relationships between user preferences and perfume recommendations.

---

## Machine Learning Models

Two classification models were trained and evaluated:

### 1. Decision Tree Classifier
A supervised learning algorithm used for classification by splitting data into decision-based branches.

### 2. Random Forest Classifier
An ensemble learning model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Model Training

The dataset was divided into:
- 80% training data
- 20% testing data

The models were trained using the processed dataset after encoding and scaling.

---

## Model Evaluation

Several evaluation metrics were used to measure the performance of the models.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

### Decision Tree Results
- Accuracy: 0.47
- Precision: 0.48
- Recall: 0.47
- F1-Score: 0.47

### Random Forest Results
- Accuracy: 0.53
- Precision: 0.58
- Recall: 0.53
- F1-Score: 0.53

The Random Forest model achieved better overall performance compared to the Decision Tree model.

---

## Confusion Matrix Analysis

Confusion matrices were used to visualize prediction performance.

### Findings
- Most correct predictions appeared along the diagonal
- Some misclassifications occurred between similar perfume categories
- Random Forest showed better classification consistency

---

## Graphical User Interface (GUI)

A Streamlit-based GUI was developed to allow users to interact with the recommendation system.

### GUI Features
- User-friendly interface
- Interactive perfume recommendation system
- Real-time predictions
- Easy input selection

The GUI makes the project more practical and accessible for users without technical knowledge.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## Files Included

- `app.py` → Main Streamlit application
- `perfume_model.pkl` → Trained machine learning model
- `encoders.pkl` → Encoders for categorical variables
- `scaler.pkl` → Feature scaling model
- `clean_perfume_data.csv` → Dataset used for training
- `README.md` → Project documentation

---

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

---

## How to Run the Project

Run the Streamlit application using:

```bash
streamlit run app.py
```

---

## Project Goal

The goal of this project is to demonstrate how machine learning can be applied to recommendation systems and personalized user experiences using real-world data and interactive applications.

---

## Future Improvements

The project can be improved in the future by:
- Using larger perfume datasets
- Applying deep learning models
- Improving recommendation accuracy
- Adding more perfume features
- Deploying the application online

---

import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Import libraries needed for splitting data and training models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Load dataset in code
df = pd.read_csv("cperfume_dataset.csv")

# Exploration dataset
# Show forst 5 rows of dataset
df.head()
# Show last 5 rows of dataset
df.tail()
# Show basic information of dataset
df.info()
# Show the number of rows and columns in the dataset
df.shape
# Display all column names
df.columns
# Show statistical summary
df.describe().round(2)
# Check the number of missing values in each column
df.isnull().sum()

# Visualization before cleaning

# # Age distribution
# plt.figure()
# df["Age"].hist()
# plt.title("Age Distribution(befor cleaning)")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.show()

# # Scent distribution
# plt.figure()
# df["Preferred_Scent"].value_counts().plot(kind="bar")
# plt.title("Scent Distribution(befor cleaning)")
# plt.xlabel("Scent")
# plt.ylabel("Frequency")
# plt.show()

# Handling null values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Weather"].fillna(df["Weather"].mode()[0], inplace=True)

# Check null values after clean
df.isnull().sum()

# Check how many duplicate rows exist in dataset
df.duplicated().sum()

# Remove duplicate rows
df = df.drop_duplicates()
df.duplicated().sum()

# Show the number of rows and columns in dataset after deleting
df.shape

# Make sure that age is in limited range
df = df[(df["Age"] >= 15) & (df["Age"] <= 60)]

# Show statistical summary after cleaning
df.describe().round(2)

# # Visualization after cleaning
# # Age distribution
# plt.figure()
# df["Age"].hist()
# plt.title("Age Distribution(after cleaning)")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.show()

# # Scent distribution
# plt.figure()
# df["Preferred_Scent"].value_counts().plot(kind="bar")
# plt.title("Scent Distribution(after cleaning)")
# plt.xlabel("Scent")
# plt.ylabel("Frequency")
# plt.show()
# print(df["Weather"].unique())
# print(df["Occasion"].unique())
# print(df["Strength"].unique())
# print(df["Preferred_Scent"].unique())

# Make a copy of the cleaned dataset
df_model = df.copy()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_model["Age"] = scaler.fit_transform(df_model[["Age"]])
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
# Create a LabelEncoder object
# LabelEncoder changes text values into numbers


encoders = {}
# These columns contain text values, so we need to encode them
categorical_columns = ['Gender', 'Weather', 'Occasion', 'Strength', 'Preferred_Scent']

# Convert each categorical column from text to numbers
for column in categorical_columns:
    encoder = LabelEncoder()
    df_model[column] = encoder.fit_transform(df_model[column])
    encoders[column] = encoder
# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
# Show first 5 rows after encoding
df_model.head()
#save dataset
df_model.to_csv("clean_perfume_data.csv", index=False)
# X contains the input features used to make predictions
X = df_model.drop('Preferred_Scent', axis=1)

# y contains the target column we want to predict
y = df_model['Preferred_Scent']



# Split the dataset into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Decision Tree Classifier model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model using the training data
decision_tree_model.fit(X_train, y_train)

print("Decision Tree model training completed.")

# Create the Random Forest Classifier model
# n_estimators=100 means the model uses 100 decision trees
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the Random Forest model using the training data
random_forest_model.fit(X_train, y_train)
#-------------

# Save model
with open("perfume_model.pkl", "wb") as file:
    pickle.dump(random_forest_model, file)
    
#-------------
print("Random Forest model training completed and saveds.")

# Show the size of training and testing data
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# This plot shows the confusion matrix for the Decision Tree model.
# It compares the actual labels with the predicted labels.
# The diagonal values represent correct predictions, while other values indicate misclassifications.

# Predictions
dt_pred = decision_tree_model.predict(X_test)
rf_pred = random_forest_model.predict(X_test)

# Accuracy
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

# # This plot shows the confusion matrix for the Random Forest model.
# # It is used to evaluate how well the model performs in classification tasks.
# # Higher values on the diagonal indicate better model performance.
# cm_dt = confusion_matrix(y_test, dt_pred)
# plt.figure(figsize=(6,4))
# ConfusionMatrixDisplay(cm_dt).plot()
# plt.title("Decision Tree Confusion Matrix")
# plt.show()

# # The confusion matrix visualizes the performance of the classification model.
# # It shows the number of correct and incorrect predictions for each class.
# cm_rf = confusion_matrix(y_test, rf_pred)
# plt.figure(figsize=(6,4))
# ConfusionMatrixDisplay(cm_rf).plot()
# plt.title("Random Forest Confusion Matrix")
# plt.show()

# # Feature importance indicates how much each feature contributes to the prediction.
# # In this model, Age appears to have the highest influence on the output.

# importance = random_forest_model.feature_importances_
# features = X.columns
# plt.figure(figsize=(7,4))
# plt.bar(features, importance)
# plt.title("Feature Importance")
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.xticks(rotation=45)
# plt.show()

"""### Model Evaluation and Testing

In this section, we evaluate the performance of the trained models using the test dataset.  
We use several evaluation metrics such as Accuracy, Precision, Recall, and F1-score  
to measure how well the models predict suitable perfume recommendations.
"""
# Decision Tree Evaluation

# Evaluate Decision Tree on test data
print("Decision Tree Evaluation")

print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, average='weighted'))
print("Recall:", recall_score(y_test, dt_pred, average='weighted'))
print("F1-score:", f1_score(y_test, dt_pred, average='weighted'))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

# Random Forest Evaluation

# Evaluate Random Forest on test data
print("Random Forest Evaluation")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, average='weighted'))
print("Recall:", recall_score(y_test, rf_pred, average='weighted'))
print("F1-score:", f1_score(y_test, rf_pred, average='weighted'))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

"""### Observation
Random Forest achieved better performance than Decision Tree in terms of accuracy and F1-score, indicating that it is more effective for this dataset.
"""
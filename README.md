# 🧹 Task 1: Data Cleaning & Preprocessing - Titanic Dataset

## 📘 Overview
Preprocessing the Titanic dataset for ML using Python.

## 🎯 Objective
- Handle missing values  
- Encode categorical data  
- Normalize features  
- Detect & remove outliers

## 📂 Dataset
- Source: [Kaggle - Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)  
- File: `train.csv`

## 🛠️ Tools
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn

## 🧪 Steps
- Load dataset  
- Fill missing Age (median), Embarked (mode)  
- Drop Cabin, Name, Ticket, PassengerId  
- Encode Sex & Embarked  
- Add FamilySize feature  
- Boxplot outliers (Age, Fare)  
- Remove Fare outliers using IQR  
- Standardize Age & Fare

## ✅ Output
Cleaned dataset ready for machine learning.



# 📊 Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset

## 🎯 Objective
Explore the Titanic dataset to understand feature distributions, relationships, and patterns using summary statistics and visualizations.

---

## 🧰 Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn

---

## 🧪 Step-by-Step Code

1. Loaded the dataset using Pandas  
2. Displayed statistical summary using `df.describe()`  
3. Calculated separate standard deviation for each numerical feature  
4. Visualized data distribution using histograms and boxplots  
5. Generated pairplots to observe feature relationships  
6. Created correlation matrix and heatmap for correlation analysis  
7. Observed trends and outliers in features like Age and Fare  
8. Interpreted visuals to derive insights (e.g., survival rate by gender/class)  

## ✅ Output  
- Clear statistical overview of dataset  
- Visual understanding of distributions, relationships, and outliers  
- Insights on key factors affecting survival



# 📈 Task 3: Linear Regression - Housing Dataset

## 📘 Overview  
Implemented Simple and Multiple Linear Regression using the Housing dataset to predict property prices.

## 🎯 Objective  
- Import and preprocess dataset  
- Split data into training and testing sets  
- Train a Linear Regression model  
- Evaluate using MAE, MSE, and R²  
- Visualize regression line and interpret coefficients  

## 📂 Dataset  
**Source:** Custom Housing Dataset  
**File:** Housing.csv  

## 🛠️ Tools  
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

## 🧪 Steps  
1. Loaded and explored the dataset  
2. Checked and cleaned null values if present  
3. Converted categorical columns like 'mainroad', 'furnishingstatus', etc. into numerical using `LabelEncoder`  
4. Split the dataset into training and testing sets using `train_test_split()`  
5. Trained a `LinearRegression` model using Scikit-learn  
6. Evaluated the model with MAE, MSE, and R² score  
7. Visualized the regression line for better interpretation  
8. Displayed and interpreted coefficients to understand feature impact  

## ✅ Output  
- A working Linear Regression model capable of predicting housing prices  
- Plotted regression line and printed evaluation metrics  
- Understood which features most influence price prediction


 

# 🧹Task 4: Classification with Logistic Regression
# 📘 Overview
In this task, I implemented a binary classification model using Logistic Regression to predict class labels based on given features.

# 🎯 Objective:
-Choose a binary classification dataset.
-Split the dataset into training and testing sets.
-Standardize the features for better model performance.
-Train a Logistic Regression model.
-Evaluate the model using key metrics like confusion matrix, precision, recall, and ROC-AUC.
-Tune the decision threshold and explain the sigmoid function.

# 📂 Dataset
File : data.csv
Source : Provided dataset for internship task.

# 🛠️ Tools Used:
~ Python
~ Pandas – Data handling and preprocessing
~ Matplotlib – Data visualization
~ Scikit-learn – Model building and evaluation

# 🧪 Steps:
1.Load the Dataset

2.Read the CSV file using Pandas.

3.Data Preprocessing

4.Handle missing values if any.

5.Ensure target variable is binary (0/1).

6.Split Dataset

7.Train/Test split using train_test_split from Scikit-learn.

8.Feature Scaling

9.Standardize numerical features using StandardScaler.

10.Model Training

11.Fit a Logistic Regression model on the training data.

12.Prediction & Evaluation

13.Predict class labels on test data.

14.Calculate metrics:

Confusion Matrix

Precision, Recall, F1-score

ROC Curve and AUC score

15.Threshold Tuning

Adjust classification threshold for better balance between precision and recall.

16.Sigmoid Function Explanation

Demonstrated how Logistic Regression uses the sigmoid function to map predictions between 0 and 1.

# ✅ Output
A trained Logistic Regression model with evaluation metrics and visualizations (confusion matrix & ROC curve).



# 📈Task 5: Decision Trees and Random Forests

## 📌 Objective
Learn and implement tree-based models for **classification** and **regression** using the Heart Disease dataset.  
This task covers:
- Training a Decision Tree Classifier
- Analyzing overfitting and controlling tree depth
- Training a Random Forest and comparing performance
- Interpreting feature importances
- Evaluating models using cross-validation

## 📂 Dataset
**Heart Disease Dataset** (`heart.csv`)  
This dataset contains patient health information and a target variable:
- **target = 1** → Patient has heart disease  
- **target = 0** → Patient does not have heart disease  

## 🛠 Tools & Libraries
- Python 
- Pandas & NumPy (Data handling)
- Scikit-learn (Model building & evaluation)
- Matplotlib & Seaborn (Visualization)
- Graphviz (Decision Tree visualization)
  
## 🚀 Steps Implemented

### 1️⃣ Data Loading & Exploration
- Loaded dataset using `pandas`
- Checked shape, missing values, and basic statistics

### 2️⃣ Train-Test Split
- Features (`X`) → all columns except `target`
- Target (`y`) → `target` column
- Split into training (80%) and testing (20%) sets

### 3️⃣ Decision Tree Classifier
- Trained a **full-depth** decision tree
- Checked accuracy on both training and testing sets

### 4️⃣ Overfitting Analysis
- Observed high training accuracy but lower testing accuracy for full tree (overfitting)
- Limited `max_depth=3` to reduce overfitting
- Compared train/test accuracies

### 5️⃣ Tree Visualization
- Visualized the limited depth Decision Tree using Graphviz

### 6️⃣ Accuracy vs Tree Depth Plot
- Created a graph to show training/testing accuracy for depths 1–10

### 7️⃣ Random Forest Classifier
- Trained with `n_estimators=100`
- Compared accuracy with Decision Tree

### 8️⃣ Feature Importance
- Visualized most important features influencing the prediction

### 9️⃣ Cross-Validation
- Performed 5-fold CV and calculated average accuracy

### 🔟 Final Evaluation
- Generated confusion matrix & classification report for Random Forest
- 
## 📊 Results Summary

| Model                          | Train Accuracy | Test Accuracy |
|--------------------------------|---------------|--------------|
| Decision Tree (Full Depth)     | ~1.00         | ~0.80        |
| Decision Tree (Limited Depth)  | ~0.87         | ~0.85        |
| Random Forest                  | ~0.99         | ~0.88        |

- **Full Depth Tree** → Overfits the training data  
- **Limited Depth Tree** → Balanced performance, less overfitting  
- **Random Forest** → Best overall performance

## 📈 Visualizations
- Decision Tree flowchart
- Accuracy vs Tree Depth graph
- Feature importance bar chart
- Confusion matrix heatmap


# Task 6: K-Nearest Neighbors (KNN) Classification

## 📌 Objective
The objective of this task is to understand and implement the **K-Nearest Neighbors (KNN)** algorithm for classification problems using the Iris dataset.


## 🛠 Tools & Libraries Used
- **Python**
- **Pandas** – Data handling
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Scikit-learn** – Machine learning model & evaluation


## 📂 Dataset
- **Name**: Iris Dataset (`Iris.csv`)
- **Description**: A classic dataset containing 150 samples of iris flowers, with 4 features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width  
  And the target class:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

## 🚀 Steps Implemented

### **Step 1: Choose Dataset & Normalize Features**
- Loaded the Iris dataset using Pandas.
- Normalized the features using **StandardScaler**.

### **Step 2: Train the Model using KNeighborsClassifier**
- Used `KNeighborsClassifier` from **scikit-learn**.
- Initially tested with a default value of `k=5`.

### **Step 3: Experiment with Different Values of K**
- Tested multiple values of `k` to find the best accuracy.
- Selected the best `k` value for final model training.

### **Step 4: Evaluate Model**
- Evaluated using:
  - **Accuracy score**
  - **Confusion matrix**
  - **Classification report**

### **Step 5: Visualize Decision Boundaries**
- Used **PCA** to reduce features to 2D for visualization.
- Plotted decision boundaries showing separation between classes.

## 📊 Results
- Best k-value: Varies based on experiment (commonly k=5)  
- Accuracy: Achieved ~97% accuracy on the test dataset.
- The decision boundary plot clearly showed class separation.

## 📷 Output Visualization
1. Confusion Matrix
2. PCA Decision Boundary plot



# Task 7: Support Vector Machines (SVM)

## 📌 Objective
The goal of this task was to **use SVMs for linear and non-linear classification** on synthetic datasets and analyze their performance using decision boundaries, cross-validation, and hyperparameter tuning.

## 🛠 Tools & Libraries Used
- **Python 3.x**
- **Scikit-learn** → Model building, pipelines, SVM training, evaluation
- **NumPy** → Numerical computations
- **Matplotlib** → Data visualization & decision boundary plotting

## 📂 Steps Performed

### 1️⃣ Load & Prepare Dataset
- Created **two binary classification datasets**:
  - **Linearly separable**: `make_blobs`
  - **Non-linear**: `make_moons`
- Split data into **training (70%)** and **testing (30%)** sets.
- Standardized features using `StandardScaler` for better SVM performance.

### 2️⃣ Train SVM Models
- **Linear SVM** for linearly separable data.
- **RBF Kernel SVM** for non-linear data patterns.
- Used **`Pipeline`** to combine scaling and classification steps.

### 3️⃣ Visualize Decision Boundaries
- Implemented a helper function to **plot decision boundaries** for both kernels.
- Showed how linear SVM creates straight hyperplanes, while RBF can curve around data.

### 4️⃣ Model Evaluation
- Evaluated on **test sets** using:
  - **Accuracy Score**
  - **Classification Report**
  - **Confusion Matrix**
- Compared performance between linear and RBF kernels.

### 5️⃣ Cross-Validation
- Applied **5-fold Stratified Cross-Validation** to estimate performance more robustly.
- Computed **mean accuracy** and **standard deviation**.

### 6️⃣ Hyperparameter Tuning
- Tuned **C** (regularization) and **gamma** (kernel coefficient) for the RBF SVM using `GridSearchCV`.
- Identified **best parameters** and improved classification accuracy.
- Visualized decision boundary for the tuned model.

## 📊 Results Summary
| Dataset       | Kernel  | Test Accuracy |
|---------------|---------|--------------|
| Blobs         | Linear  | ~99%         |
| Moons         | RBF     | ~98% (tuned) |

- **Linear SVM** performed excellently on linearly separable data.
- **RBF SVM** captured complex, non-linear boundaries effectively.




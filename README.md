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

# Fare Grouping
df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=["Low", "Medium", "High", "Very High"])
sns.countplot(data=df, x='FareGroup', hue='Survived')
plt.title("Survival by Fare Range")
plt.show()

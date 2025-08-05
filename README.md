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

## 📁 Structure




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

```python
Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# **Step 1: Load Dataset and Generate Summary Statistics**


df = pd.read_csv("Titanic-Dataset.csv.csv")

# Summary Statistics
print("Summary Statistics:\n")
print(df.describe(include='all').transpose())

# Median (not included in describe by default)
print("\nMedian of Numeric Columns:\n")
print(df.median(numeric_only=True))

# Standard Deviation
print("\nStandard Deviation of Numeric Columns:\n")
print(df.std(numeric_only=True))


# **Step 2: Create Histograms and Boxplots for Numeric Features**


# Histogram - Age
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Boxplot - Age
sns.boxplot(y=df['Age'])
plt.title("Boxplot of Age")
plt.show()

# Histogram - Fare
sns.histplot(df['Fare'].dropna(), bins=30, kde=True)
plt.title("Histogram of Fare")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# Boxplot - Fare
sns.boxplot(y=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()


# **Step 3: Use Pairplot and Correlation Matrix for Relationships**


# Encode categorical for pairplot
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Drop missing values for clean plotting
features = ['Survived', 'Pclass', 'Age', 'Fare', 'Sex', 'SibSp', 'Parch']
df_clean = df[features].dropna()

# Pairplot
sns.pairplot(df_clean, hue='Survived', diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# **Step 4: Identify Patterns, Trends, or Anomalies**


# Survival by Gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival by Gender (0=Male, 1=Female)")
plt.show()

# Survival by Class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival by Class")
plt.show()

# Survival by Embarked
sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title("Survival by Embarked Location")
plt.show()


# **Step 5: Make Basic Feature-Level Inferences from Visuals**


# Age Grouping
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])
sns.countplot(data=df, x='AgeGroup', hue='Survived')
plt.title("Survival by Age Group")
plt.show()

# Fare Grouping
df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=["Low", "Medium", "High", "Very High"])
sns.countplot(data=df, x='FareGroup', hue='Survived')
plt.title("Survival by Fare Range")
plt.show()

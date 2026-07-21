# 🩺 Predicting Student Health Risk

A complete end-to-end Machine Learning solution for the **Kaggle Playground Series - Season 6 Episode 7** competition.

## 📌 Overview

The objective is to classify students into one of three health conditions:

- 🟢 Fit
- 🟡 Unhealthy
- 🔴 At-Risk

The competition is evaluated using **Balanced Accuracy**.

---

## 📊 Dataset

Training Samples: **690,088**

Test Samples: **295,753**

Features:

- Sleep Duration
- Heart Rate
- BMI
- Calorie Expenditure
- Step Count
- Exercise Duration
- Water Intake
- Diet Type
- Stress Level
- Sleep Quality
- Physical Activity Level
- Smoking & Alcohol
- Gender

---

## 🔍 Exploratory Data Analysis

Performed:

- Missing value analysis
- Class imbalance visualization
- Feature inspection
- Numerical statistics
- Categorical feature analysis

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

- Median Imputation
- Most Frequent Imputation
- One-Hot Encoding
- Label Encoding

### Models

✅ Random Forest

✅ CatBoost

### CatBoost Improvements

- Balanced Class Weights
- Early Stopping
- Tuned Hyperparameters
- Native Categorical Feature Handling

---

## 🏆 Results

| Model | Public Score |
|--------|-------------:|
| Random Forest | 0.84472 |
| CatBoost | 0.84939 |
| **Tuned CatBoost** | **0.89986** |

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- CatBoost

---

## 📁 Project Structure

```text
data/
images/
notebooks/
README.md
requirements.txt
```

---

## 👩‍💻 Author

**Ayesha Asif**

BS Data Science

University of Engineering and Technology (UET) Lahore

GitHub:
https://github.com/ayeshaasif12089-ship-it

Kaggle:
https://www.kaggle.com/ayeshaasif1512

LinkedIn:
https://www.linkedin.com/in/ayeshaasif5420
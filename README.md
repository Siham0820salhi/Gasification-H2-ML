# 🔬 Hydrogen Volumetric Percentage Prediction Using Machine Learning.

## 📌 Project Overview

Hydrogen is considered one of the most promising clean energy sources for the future. Accurately predicting hydrogen production during the gasification process is essential for improving energy efficiency and optimizing operating conditions.

This project presents a complete Machine Learning pipeline for predicting the **hydrogen volumetric percentage (H₂ vol%)** generated during biomass gasification. The workflow includes data preprocessing, exploratory data analysis, feature engineering, model development, model comparison, explainability analysis, and deployment through a Streamlit web application.

---

## 👥 Project Members

- **Siham Salhi**
- **Ibtissam Essadiki**

---

## 🎯 Project Objectives

- Predict the hydrogen volumetric percentage (H₂ vol%) produced during biomass gasification.
- Analyze the influence of process parameters on hydrogen production.
- Develop and compare several Machine Learning regression models.
- Improve prediction accuracy through feature engineering.
- Interpret model predictions using SHAP Explainability.
- Deploy an interactive prediction application using Streamlit.

---

## 📊 Dataset Description

The dataset contains experimental gasification data collected under different operating conditions.

### Input Features

- Gasification Temperature
- Process Time
- Gasifying Agent Flow Rate
- Catalyst Ratio
- Catalyst Type
- Sample Type
- Gasifying Agent Type
- Other process parameters

### Target Variable

- **Hydrogen Volumetric Percentage (H₂ vol%)**

---

## ⚙️ Methodology

### 1. Data Preprocessing

- Data cleaning
- Missing value verification
- Duplicate removal
- Data formatting

Notebook:
```
01_nettoyage.ipynb
```

---

### 2. Exploratory Data Analysis (EDA)

Performed statistical and graphical analyses to better understand the dataset.

- Distribution analysis
- Correlation analysis
- Outlier detection
- Feature relationships

Notebook:
```
02_eda.ipynb
```

---

### 3. Feature Engineering

Several new features were created to improve model performance.

Examples include:

- Cumul_Temp
- Thermal_Work
- Temp_squared

Notebook:
```
03_feature_engineering.ipynb
```

---

### 4. Machine Learning Models

Three regression models were developed and evaluated.

| Model | Description |
|--------|-------------|
| Random Forest | Ensemble decision trees |
| Extra Trees | Extremely Randomized Trees |
| XGBoost | Gradient Boosting algorithm |

Notebooks:

```
04_random_forest.ipynb
05_extra_trees.ipynb
06_xgboost.ipynb
```

---

### 5. Model Comparison

The three models were evaluated using multiple regression metrics.

Notebook:

```
07_comparaison.ipynb
```

---

### 6. Model Explainability

SHAP (SHapley Additive Explanations) was used to interpret the trained models and identify the most influential features affecting hydrogen prediction.

Notebook:

```
08_shap.ipynb
```

---

## 📈 Model Performance

| Model | R² Score | MAE | RMSE |
|--------|---------:|---------:|---------:|
| ⭐ Extra Trees | **0.8286** | **3.095** | **4.556** |
| Random Forest | 0.8205 | 3.191 | 4.662 |
| XGBoost | 0.7990 | 3.391 | 4.933 |

### 🏆 Best Performing Model

**Extra Trees Regressor**

- Highest prediction accuracy
- Lowest prediction error
- Best overall generalization performance

---

## 🔍 Feature Importance (SHAP)

The SHAP analysis revealed that the most influential engineered features are:

- Cumul_Temp
- Thermal_Work
- Temp_squared

These features contribute significantly to improving hydrogen prediction accuracy.

---

## 💻 Streamlit Application

An interactive Streamlit application was developed to allow users to estimate hydrogen production by entering gasification process parameters.

Main features:

- User-friendly interface
- Real-time prediction
- Automatic model selection
- Prediction history
- Input validation

Run the application:

```bash
streamlit run app_with_history.py
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📂 Project Structure

```
Gasification-H2-ML
│
├── 01_nettoyage.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_random_forest.ipynb
├── 05_extra_trees.ipynb
├── 06_xgboost.ipynb
├── 07_comparaison.ipynb
├── 08_shap.ipynb
├── train_models_separated.py
├── app_with_history.py
├── models/
├── data_clean.csv
├── Gasification Dataset.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Siham0820salhi/Gasification-H2-ML.git
```

Move into the project directory

```bash
cd Gasification-H2-ML
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit application:

```bash
streamlit run app_with_history.py
```

The application will be available at:

```
http://localhost:8501
```

---

## 📑 Workflow

```
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Comparison
        │
        ▼
SHAP Explainability
        │
        ▼
Streamlit Deployment
```

---

## 🔮 Future Improvements

- Deep Learning models for hydrogen prediction.
- Hyperparameter optimization using Bayesian Optimization.
- Real-time prediction using IoT sensor data.
- Deployment on cloud platforms.
- Integration of additional gasification datasets.

---

## 📄 Project Report

A detailed project report describing the methodology, experiments, results, and conclusions is included in this repository.

---

## 📬 Contact

**Siham Salhi**

GitHub: **https://github.com/Siham0820salhi**

Email: **salhisiham665@gmail.com**

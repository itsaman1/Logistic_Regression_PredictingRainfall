# 🌧️ Rainfall Prediction using Logistic Regression

## 📌 Project Overview
This project applies **Logistic Regression** to predict whether it will rain tomorrow based on historical weather data from Australia. The dataset includes **10 years of daily weather observations** from various weather stations. The objective is to build a binary classification model that predicts `RainTomorrow` (Yes/No) based on meteorological features such as temperature, humidity, wind speed, and pressure.

## 📂 Dataset
- **Source:** [Rain in Australia - Kaggle](https://kaggle.com/jsphyg/weather-dataset-rattle-package)
- **Target Variable:** `RainTomorrow`
- **Features:** Various weather-related attributes (humidity, wind speed, pressure, temperature, etc.)

Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 

- we take linear combination (or weighted sum of the input features) 
- we apply the sigmoid function to the result to obtain a number between 0 and 1
- this number represents the probability of the input being classified as "Yes"
- instead of RMSE, the cross entropy loss function is used to evaluate the results


Here's a visual summary of how a logistic regression model is structured ([source](http://datahacker.rs/005-pytorch-logistic-regression-in-pytorch/)):


<img src="https://i.imgur.com/YMaMo5D.png" width="480">

The sigmoid function applied to the linear combination of inputs has the following formula:

<img src="https://i.imgur.com/sAVwvZP.png" width="400">


## 🔍 Steps Involved

### 1️⃣ Data Preprocessing
- **Handling missing values** and ensuring data integrity.
- **Exploratory Data Analysis (EDA)** to understand patterns and correlations.
- **Feature selection & engineering** to retain the most relevant attributes.

### 2️⃣ Model Training
- **Encoding categorical variables** (e.g., wind direction, location) using One-Hot Encoding.
- **Standardizing numerical features** to improve model performance.
- Training a **Logistic Regression model** using `scikit-learn` for binary classification.

### 3️⃣ Model Evaluation
- **Performance metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- **ROC Curve & AUC Score** for assessing classification performance.
- Identified **humidity and wind speed** as the strongest indicators of rainfall.

## 🛠️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rainfall-prediction.git
   cd rainfall-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook LogisticRegression.ipynb
   ```

## 📈 Key Insights
- **Humidity levels** emerged as the most important predictor of rainfall.
- The model provides **valuable insights for weather forecasting applications**.
- Future improvements can be made using **ensemble learning techniques** (Random Forest, XGBoost) for better accuracy.

## 📌 Why This Matters?
Accurate rainfall prediction can help **farmers, meteorologists, and city planners** make data-driven decisions and prepare for extreme weather conditions.

## 🚀 Future Scope
- Exploring **advanced ML models** like Random Forest, SVM, and Neural Networks.
- Fine-tuning hyperparameters to enhance model performance.
- Deploying the model as a web application using Flask/Streamlit.

## 🔗 References
- Kaggle Dataset: [Rain in Australia](https://kaggle.com/jsphyg/weather-dataset-rattle-package)
- Scikit-learn Documentation: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

📢 **Feel free to fork the repository and contribute!** Let's connect and discuss **Machine Learning, AI, and Data Science!** 🚀


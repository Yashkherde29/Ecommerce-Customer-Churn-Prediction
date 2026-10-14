# 🛒 E-Commerce Customer Churn Prediction

An end-to-end Machine Learning project that analyzes e-commerce customer behavior and predicts customer churn through an interactive Flask web application.

## 📌 Project Overview

Customer churn occurs when customers stop purchasing from a business. Predicting churn helps businesses identify customers who may be at risk of leaving and develop data-driven retention strategies.

This project combines customer analytics, data visualization, and Machine Learning to provide an interactive platform for exploring customer behavior and predicting churn.

## ✨ Key Features

- **Customer Churn Prediction:** Predicts whether a customer is likely to churn.
- **Churn Probability:** Displays the model's estimated probability of churn.
- **Risk Classification:** Categorizes predicted churn risk as Low, Medium, or High.
- **Interactive Analytics Dashboard:** Displays customer statistics, spending patterns, orders, and churn metrics.
- **Data Visualization:** Explores trends across product categories, payment methods, and customer behavior.
- **User-Friendly Interface:** Provides an interactive web experience using Flask.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, CSS, JavaScript

## 📊 Dataset

The project uses an e-commerce customer dataset containing approximately 100,000 customer records and 15 columns.

- **Target Variable:** `Churned`
- **Target Values:** Yes / No
- **Purpose:** Analyze customer behavior and train a classification model to predict churn.

## 📂 Project Structure

```text
Ecommerce-Customer-Churn-Prediction/
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── data/
│   └── ecommerce.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── ecommerce_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualization.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

*This is the intended structure; adjust it to match the files in your actual project.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd Ecommerce-Customer-Churn-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

From the project root, run the command that matches your actual file location:

```bash
python app/app.py
```

Open the local URL shown in the terminal, typically:

`http://127.0.0.1:5000`

## 🎯 Project Objectives

- Apply Machine Learning to customer churn prediction.
- Analyze customer behavior using exploratory data analysis.
- Build an interactive analytics dashboard.
- Convert model predictions into understandable risk categories.
- Demonstrate an end-to-end Python and Flask ML application.

## 🎥 Project Demo

A demonstration video showcasing the application is available in the repository.

## 👨‍💻 Author

**Yash Kherde**

B.E. Data Science Student | Machine Learning | Python | Data Analytics

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Predictions depend on the dataset and trained model and should not be treated as guaranteed outcomes.

## ⭐ Support

If you find this project useful, consider giving the repository a star!

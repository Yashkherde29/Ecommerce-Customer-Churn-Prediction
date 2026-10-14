# E-Commerce Customer Churn Prediction

A machine-learning web application that predicts whether an e-commerce customer is likely to churn.

## Dataset

The application uses `ecommerce.csv`.

Dataset size:

- 100,000 customers
- 15 columns

## Prediction Target

The primary target is:

`Churned`

The model predicts:

- Churn: Yes / No
- Churn probability
- Risk level

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- HTML
- CSS
- JavaScript
- Matplotlib
- Seaborn

## Project Structure

```text
Ecommerce-Prediction/
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
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
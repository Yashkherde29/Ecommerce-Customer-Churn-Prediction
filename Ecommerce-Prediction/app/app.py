from pathlib import Path
import sys

import pandas as pd
from flask import Flask, render_template, request


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR / "src"))

from predict import predict_churn


# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

DATA_PATH = BASE_DIR / "data" / "ecommerce.csv"

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    total_customers = len(df)

    total_orders = int(
        df["Total_Orders"].sum()
    )

    total_spent = float(
        df["Total_Spent"].sum()
    )

    average_spent = float(
        df["Total_Spent"].mean()
    )

    average_order_value = float(
        df["Average_Order_Value"].mean()
    )

    churned_customers = int(
        (df["Churned"].str.lower() == "yes").sum()
    )

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    categories = (
        df["Preferred_Category"]
        .value_counts()
        .to_dict()
    )

    payment_methods = (
        df["Payment_Method"]
        .value_counts()
        .to_dict()
    )

    return render_template(
        "dashboard.html",
        total_customers=total_customers,
        total_orders=total_orders,
        total_spent=total_spent,
        average_spent=average_spent,
        average_order_value=average_order_value,
        churned_customers=churned_customers,
        churn_rate=round(churn_rate, 2),
        categories=categories,
        payment_methods=payment_methods
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    result = None
    error = None

    if request.method == "POST":

        try:

            customer_data = {
                "Age": int(request.form["Age"]),

                "Gender": request.form["Gender"],

                "Location": request.form["Location"],

                "Signup_Date": request.form["Signup_Date"],

                "Last_Login": request.form["Last_Login"],

                "Total_Spent": float(
                    request.form["Total_Spent"]
                ),

                "Total_Orders": int(
                    request.form["Total_Orders"]
                ),

                "Average_Order_Value": float(
                    request.form["Average_Order_Value"]
                ),

                "Preferred_Category":
                    request.form["Preferred_Category"],

                "Payment_Method":
                    request.form["Payment_Method"],

                "Discount_Used (%)": float(
                    request.form["Discount_Used"]
                ),

                "Time_Spent_on_App (mins)": float(
                    request.form["Time_Spent"]
                ),

                "Loyalty_Member":
                    request.form["Loyalty_Member"]
            }

            result = predict_churn(
                customer_data
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "prediction.html",
        result=result,
        error=error
    )


# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
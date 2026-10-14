from pathlib import Path

import joblib
import pandas as pd

from preprocessing import prepare_features


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():
    """Load the trained machine learning pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Please run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# PREDICT CHURN
# --------------------------------------------------

def predict_churn(customer_data):
    """
    Predict customer churn.

    customer_data:
        Dictionary containing raw customer information.
    """

    model = load_model()

    # Convert dictionary to DataFrame
    data = pd.DataFrame([customer_data])

    # IMPORTANT:
    # Apply the exact same feature engineering
    # that was used during model training.
    data = prepare_features(data)

    # Make prediction
    prediction = model.predict(data)[0]

    # Get churn probability
    probability = model.predict_proba(data)[0][1]

    # Determine risk level
    if probability < 0.30:
        risk = "Low"

    elif probability < 0.70:
        risk = "Medium"

    else:
        risk = "High"

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": round(float(probability) * 100, 2),
        "risk": risk
    }


# --------------------------------------------------
# TEST PREDICTION
# --------------------------------------------------

if __name__ == "__main__":

    sample_customer = {

        "Age": 25,

        "Gender": "Male",

        "Location": "Toronto",

        "Signup_Date": "29-07-2022",

        "Last_Login": "24-10-2022",

        "Total_Spent": 1921.08,

        "Total_Orders": 25,

        "Average_Order_Value": 91.48,

        "Preferred_Category": "Clothing",

        "Payment_Method": "Credit Card",

        "Discount_Used (%)": 16,

        "Time_Spent_on_App (mins)": 48.29,

        "Loyalty_Member": "Yes"
    }

    result = predict_churn(sample_customer)

    print("\nPrediction Result")
    print("-" * 30)

    print(
        f"Churn Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"Probability: "
        f"{result['probability']}%"
    )

    print(
        f"Risk Level: "
        f"{result['risk']}"
    )
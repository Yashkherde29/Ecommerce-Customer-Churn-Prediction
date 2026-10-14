import pandas as pd


TARGET_COLUMN = "Churned"

DROP_COLUMNS = [
    "Customer_ID",
]

DATE_COLUMNS = [
    "Signup_Date",
    "Last_Login",
]


def load_data(file_path):
    """Load the ecommerce dataset."""
    return pd.read_csv(file_path)


def prepare_features(df):
    """
    Prepare raw ecommerce data for machine learning.

    Date columns are converted into useful numerical features.
    Customer_ID is removed because it is only an identifier.
    """

    data = df.copy()

    # Remove target column if it exists
    if TARGET_COLUMN in data.columns:
        data = data.drop(columns=[TARGET_COLUMN])

    # Remove identifier
    for column in DROP_COLUMNS:
        if column in data.columns:
            data = data.drop(columns=[column])

    # Convert date columns
    for column in DATE_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_datetime(
                data[column],
                format="%d-%m-%Y",
                errors="coerce"
            )

            data[f"{column}_Year"] = data[column].dt.year
            data[f"{column}_Month"] = data[column].dt.month
            data[f"{column}_Day"] = data[column].dt.day
            data[f"{column}_DayOfWeek"] = data[column].dt.dayofweek

            data = data.drop(columns=[column])

    return data


def prepare_target(df):
    """Convert Churned Yes/No into 1/0."""

    target = df[TARGET_COLUMN].copy()

    target = (
        target
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0
        })
    )

    if target.isna().any():
        raise ValueError(
            "Unexpected values found in Churned column."
        )

    return target.astype(int)
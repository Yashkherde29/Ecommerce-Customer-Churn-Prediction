import matplotlib.pyplot as plt
import seaborn as sns


def plot_churn_distribution(df):
    """Plot churn distribution."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Churned"
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churned")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()


def plot_category_distribution(df):
    """Plot preferred product categories."""

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        y="Preferred_Category",
        order=df["Preferred_Category"].value_counts().index
    )

    plt.title("Preferred Product Categories")
    plt.xlabel("Number of Customers")
    plt.ylabel("Category")

    plt.tight_layout()
    plt.show()


def plot_spending_distribution(df):
    """Plot customer spending distribution."""

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Total_Spent",
        bins=40,
        kde=True
    )

    plt.title("Customer Spending Distribution")
    plt.xlabel("Total Spent")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()


def plot_loyalty_churn(df):
    """Compare loyalty membership and churn."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Loyalty_Member",
        hue="Churned"
    )

    plt.title("Loyalty Membership vs Churn")
    plt.xlabel("Loyalty Member")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()
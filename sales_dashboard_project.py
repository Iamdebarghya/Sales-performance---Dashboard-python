
# Sales Performance Dashboard Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Data Preprocessing and Analysis
def data_analysis(data):
    st.title("Sales Performance Dashboard")
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.write("Dataset Summary:")
    st.write(data.describe())
    st.write("Missing Values:")
    st.write(data.isnull().sum())

    # Visualization: Sales by Region
    st.subheader("Sales by Region")
    region_sales = data.groupby("Region")["Sales"].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x="Region", y="Sales", data=region_sales, ax=ax)
    st.pyplot(fig)

    # Visualization: Product-wise Sales
    st.subheader("Product-wise Sales")
    product_sales = data.groupby("Product")["Sales"].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x="Product", y="Sales", data=product_sales, ax=ax)
    st.pyplot(fig)

# Predictive Analysis
def predictive_analysis(data):
    st.subheader("Predictive Analysis: Sales Forecasting")
    st.write("This section demonstrates a simple linear regression model to predict sales.")

    # Feature Engineering
    data['Day'] = pd.to_datetime(data['Date']).dt.day
    X = data[['Day']]
    y = data['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Visualization of Predictions
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)

# Main Function
def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Data Analysis", "Predictive Analysis"])

    if option == "Data Analysis":
        data_analysis(data)
    elif option == "Predictive Analysis":
        predictive_analysis(data)

if __name__ == "__main__":
    main()

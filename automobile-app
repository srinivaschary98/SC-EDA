import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app config
st.set_page_config(page_title="UCI Auto Regression", layout="centered")

st.title("🚗 UCI Auto MPG Regression App")
st.markdown("""
Analyze automobile data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data) using **multiple linear regression**.
""")

# Load dataset from UCI
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    columns = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
        "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width",
        "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
        "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
        "city-mpg", "highway-mpg", "price"
    ]
    df = pd.read_csv(url, names=columns, na_values='?')
    return df

df = load_data()

# Clean data
numeric_cols = ['price', 'horsepower', 'engine-size', 'curb-weight', 'wheel-base', 'length', 'width', 'height', 'compression-ratio', 'peak-rpm', 'city-mpg', 'highway-mpg', 'bore', 'stroke']
df.dropna(subset=numeric_cols, inplace=True)
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Variable selection
st.subheader("📌 Select Variables for Regression")
all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
y_var = st.selectbox("Dependent Variable (y)", all_cols)
x_vars = st.multiselect("Independent Variables (X)", [col for col in all_cols if col != y_var])

# Regression
if x_vars:
    X = df[x_vars]
    X = sm.add_constant(X)
    y = df[y_var]

    model = sm.OLS(y, X).fit()

    st.subheader("📈 Regression Summary")
    st.text(model.summary())

    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[x_vars + [y_var]].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.info("Please select at least one independent variable to run the regression.")

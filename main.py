import os
import pandas as pd
import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Housing prices prediction")


def generate_data():
    df = pd.read_csv("realty_data.csv")
    data = df[["price", "floor", "total_square"]]
    return data


def train_model(data):
    X = data[["total_square", "floor"]]
    y = data["price"]
    model = LinearRegression()
    model.fit(X, y)
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def read_model(path="linear_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


model_path = "linear_model.pkl"

if not os.path.exists(model_path):
    st.info("training model, please wait...")
    data = generate_data()
    model = train_model(data)
    st.success("Model trained and saved!")
else:
    model = read_model(model_path)


st.title("🏠 Housing prices prediction")

st.sidebar.header("Enter flat's parameters")
area = st.sidebar.number_input("Total area:", min_value=10, max_value=300, value=50)
floor = st.sidebar.number_input("Floor:", min_value=1, max_value=50, value=5)

if st.sidebar.button("Calculate the price"):
    input_df = pd.DataFrame({"total_square": [area], "floor": [floor]})
    pred = model.predict(input_df)[0]
    st.subheader("Predicted flat price:")
    st.write(f"**{pred:,.0f} ₽**")
else:
    st.write("Enter the parameters and press the button")

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Iris Flower Classifier")
st.title("Iris Flower Classifier")
st.write("Enter the flower measurements in centimeters, then click Predict.")

iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, value=0.2, step=0.1)

if st.button("Predict"):
    measurements = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(measurements)
    flower_name = iris.target_names[prediction[0]].title()
    st.success(f"Predicted flower category: {flower_name}")
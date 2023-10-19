import streamlit as st
from joblib import load
from PIL import Image

model = load('data/iris_model.joblib')

st.title('Iris Prediction')
sepal_col, petal_col = st.columns(2)

with sepal_col:
    sepal_l = st.number_input("Sepal length", min_value=0.1, max_value=15.0, value=7.5, step=0.1)
    sepal_w = st.number_input("Sepal width", min_value=0.1, max_value=15.0, value=7.5, step=0.1)
with petal_col:
    petal_l = st.number_input("Petal length", min_value=0.1, max_value=15.0, value=7.5, step=0.1)
    petal_w = st.number_input("Petal width", min_value=0.1, max_value=15.0, value=7.5, step=0.1)

col1, col2, col3 = st.columns(3)

if st.button('Make a prediction!'):
    prediction = model.predict([[sepal_l, sepal_w, petal_l, petal_w]])[0]
    if prediction == 0:
        flower_name = 'iris setosa'
        flower_pic = Image.open('images/iris_setosa.png')
    elif prediction == 1:
        flower_name = 'iris versicolour'
        flower_pic = Image.open('images/iris_versicolour.png')
    elif prediction == 2:
        flower_name = 'iris virginica'
        flower_pic = Image.open('images/iris_virginica.png')
    with col2:
        st.write("Predicted flower: ", flower_name)
        st.image(flower_pic)



import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

#STEP1 : IMPORT THE TRAINED MODEL PIPELINE
# import the trained model
model=load_model('Final_model')


# STEP2: GET NEW DATA FOR PREDICTION FROM THE FRONT END
st.title("App to predict the flower species")
sepal_length=st.slider('sepal length (cm)', 2.00, 8.00)
sepal_width=st.slider('sepal width (cm)',2.00, 4.50)
petal_length=st.slider('petal length (cm)', 1.00, 7.00)
petal_width=st.slider('petal width (cm)',0.00, 2.50)

data={
    'sepal length (cm)': sepal_length,
    'sepal width (cm)': sepal_width,
    'petal length (cm)': petal_length,
    'petal width (cm)':petal_width
}

input_data=pd.DataFrame([data])
#input_data=str(input_data)

# STEP3 : GET THE PREDICTION AND DISPLAY IT
if st.button("Predict"):
    prediction=predict_model(model, input_data)
    st.success("The predicted species of the flower is as below")
    prediction['Label'][0]

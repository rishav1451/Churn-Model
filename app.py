import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

model = load_model('churn_model.h5')
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

st.title('Customer Churn Prediction')


# User input
geography = st.selectbox('Geography', ['France','Spain','Germany'])
gender = st.selectbox('Gender', ['Male','Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# input_data = {
#     'CreditScore': 600,
#     'Geography': 'France',
#     'Gender': 'Male',
#     'Age': 40,
#     'Tenure': 3,
#     'Balance': 60000,
#     'NumOfProducts': 2,
#     'HasCrCard': 1,
#     'IsActiveMember': 1,
#     'EstimatedSalary': 50000
# }

# Prepare the input data
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_data_df = pd.DataFrame([input_data])
input_data_processed = preprocessor.transform(input_data_df)
prediction = model.predict(input_data_processed)

st.subheader('Prediction')
if prediction[0][0] > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is likely to stay.')

st.subheader('Prediction Probability')
st.write(f'Probability of staying: {100 * (1 - prediction[0][0]):.2f}%')
st.write(f'Probability of churning: {100 * prediction[0][0]:.2f}%')


st.subheader('About')
st.write('This app predicts customer churn using a pre-trained ANN model. Built with Streamlit and TensorFlow/Keras.')
st.write('Developed by Rishav Raj.')


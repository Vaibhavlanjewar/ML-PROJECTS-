import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open("knn_model.pkl", "rb"))

# Function to make predictions
def predict(college_data):
    prediction = loaded_model.predict(college_data)
    return prediction

def main():
    st.title("College Predictor")

    st.write("Enter the details below to predict college admissions:")

    # Input fields
    category = st.selectbox("Category", ['GEN', 'OBC', 'SC', 'ST'])
    quota = st.selectbox("Quota", ['AI', 'HS', 'OS', 'GO'])
    pool = st.selectbox("Pool", ['Gender-Neutral', 'Female-Only'])
    institute_type = st.selectbox("Institute Type", ['IIT', 'NIT'])
    round_no = st.slider("Round Number", min_value=0, max_value=10, value=5)
    opening_rank = st.slider("Opening Rank", min_value=0, max_value=10000, value=5000)
    closing_rank = st.slider("Closing Rank", min_value=0, max_value=10000, value=5000)

    # Convert categorical inputs to encoded form
    if category == 'GEN':
        category = 0
    elif category == 'OBC':
        category = 1
    elif category == 'SC':
        category = 2
    elif category == 'ST':
        category = 3

    if quota == 'AI':
        quota = 0
    elif quota == 'HS':
        quota = 1
    elif quota == 'OS':
        quota = 2
    elif quota == 'GO':
        quota = 3

    if pool == 'Gender-Neutral':
        pool = 0
    elif pool == 'Female-Only':
        pool = 1

    if institute_type == 'IIT':
        institute_type = 0
    elif institute_type == 'NIT':
        institute_type = 1

    # Predict button
    if st.button("Predict"):
        # Prepare data for prediction
        college_data = np.array([[category, quota, pool, institute_type, round_no, opening_rank, closing_rank]])

        # Make prediction
        prediction = predict(college_data)

        # Display prediction
        st.subheader("Predicted Result:")
        st.write(prediction)


if __name__ == "__main__":
    main()

st.text("\n---------------------------------- |Made by - VNBL|--------------------------------------------------------")
st.subheader("Contact Information")
# st.write("Made by: VNBL")
st.write("GitHub: [Vaibhavlanjewar](https://github.com/Vaibhavlanjewar/)")
st.write("LinkedIn: [Vaibhav Lanjewar](https://www.linkedin.com/in/vaibhavlanjewar/)")
st.write("--------------------------------------------------------------------------------------------------------------")



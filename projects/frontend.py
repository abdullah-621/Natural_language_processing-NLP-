import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.title("This is sentiment analysis App")

from_comment = st.radio("Select option : ", ['Own comment', 'From Data'])

if from_comment == 'Own comment':
  comment = st.text_input("Enter your comment")
else:
  file = st.file_uploader("Upload your csv files", type=['csv'])

  if file:
    df = pd.read_csv(file)
    pick_comment = df.sample(1)
    comment = pick_comment['review'].iloc[0]

    st.subheader("Comment picked : ")
    st.write(f"{comment}")

if st.button("Predict"):

  input_data = {
    "comment" : comment
  }

  try:
    response = requests.post(API_URL, json=input_data)

    result = response.json()

    if response.status_code == 200 and "prediction" in result:
      st.success(f"Sentiment : {result['prediction']}")
    else:
      st.error(f"API Error : {response.status_code}")
      st.write(result)
  except requests.exceptions.ConnectionError:
    st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
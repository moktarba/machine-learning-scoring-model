import numpy as np
import pandas as pd
import streamlit as st
import requests
import joblib
import lime
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from IPython.display import HTML


data = pd.read_csv("./data.csv")
interpretability = joblib.load("interpretability.pkl")
list_index = list(data.index)
FAST_URI = "http://localhost:8000/predict"

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'index': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

text = "Bienvenue dans le dashboard de prédiction de la sovavbilité des clients"
st.markdown(text)

choice = st.sidebar.selectbox('choisir client', list_index)
if choice != None :
    client = data.iloc[int(choice)]
    st.write(client)
    predict_btn = st.button('Prédire')
    if predict_btn:
        response = request_prediction(FAST_URI, int(choice))
        prediction = response["prediction"]
        probability = response["probability"]
        st.write('Le client est : '+prediction+" avec une probabilté de "+str(probability)[:4]);

    affiche_interpretability =  st.button("Afficher Interprétabilté")
    if affiche_interpretability :
        inter = interpretability[int(choice)]
        components.html(inter.as_html(), height=800)

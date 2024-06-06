import os
import sys
import pandas as pd
import streamlit as st
#m = st.markdown("""
#<style>
#div.stButton > button:first-child {
#    background-color: #0099ff;
#    color:#ffffff;
#}
#div.stButton > button:hover {
#    background-color: #4FAFA8;
#    color:#;
#    }
#</style>""", unsafe_allow_html=True)


sys.path.append("./code")

from tools import list_models
from predict import make_prediction

TITLE_TEMPLATE = "<h4 style='text-align: center;font-weight: bold'>{}</h4>"

models = list_models("./data/models")

st.markdown(TITLE_TEMPLATE.format("1- Choose Model"), unsafe_allow_html=True)
with st.container(border = True):
    model_selected = st.selectbox("Model", options = models, label_visibility = "collapsed")

st.write("#")
st.markdown(TITLE_TEMPLATE.format("2- Upload Data"), unsafe_allow_html=True)
with st.container(border = True):
    uploaded_file = st.file_uploader("Upload Dataset", type={"csv", "txt"}, label_visibility = "collapsed")

if uploaded_file:
    st.write("#")
    st.markdown(TITLE_TEMPLATE.format("3- Make Prediction"), unsafe_allow_html=True)
    with st.container(border = True):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict_btn = st.button("Make prediction")
        with col2:
            return_input = st.toggle("Return Input data", value = True)
        with col3:
            return_proba = st.toggle("Return Probability", value = True)

    if predict_btn:
        if uploaded_file is not None:
            with st.container(border = True):
                with st.spinner('Running...'):
                    df = pd.read_csv(uploaded_file)
                    model_path = os.path.join("./data/models", model_selected)
                    df_pred = make_prediction(df, model_path, return_input, return_proba)
                    st.write(df_pred)

            #with st.container(border = True):
            #    df_pred["PREDICTION"] = df_pred["PREDICTION"].astype("str")
            #    st.scatter_chart(
            #        df_pred,
            #        x=df_pred.columns[3],
            #        y=df_pred.columns[4],
            #        color='PREDICTION')

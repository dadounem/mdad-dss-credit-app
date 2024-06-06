import os
import sys
import pandas as pd
import numpy as np
from pickle import load
import streamlit as st
from streamlit_shap import st_shap

import shap
import xgboost
import matplotlib.pyplot as plt
import plotly.express as px 

import lightgbm as lgb

shap.initjs()

sys.path.append("./code")
from tools import list_models
from predict import make_prediction

st.set_page_config(layout="wide")


TITLE_TEMPLATE = "<h4 style='text-align: center;font-weight: bold'>{}</h4>"

models = list_models("./data/models")

st.markdown(TITLE_TEMPLATE.format("1- Choose Model"), unsafe_allow_html=True)
with st.container(border = True):    
    model_selected = st.selectbox("Model", options = models, label_visibility = "collapsed")

if model_selected:
    model_path = "./data/models/{}".format(model_selected)
    model_gs = load(open(os.path.join(model_path, "model_grid_search.pkl"), 'rb'))
    model = model_gs.best_estimator_
    features = model.feature_name_


st.write("#")
st.markdown(TITLE_TEMPLATE.format("2- Upload Data to get SHAPE Insights"), unsafe_allow_html=True)
with st.container(border = True):
    uploaded_file = st.file_uploader("Dataset", type={"csv", "txt"}, label_visibility = "collapsed")


if uploaded_file:
    st.write("#")
    st.markdown(TITLE_TEMPLATE.format("3- SHAPE Insights"), unsafe_allow_html=True)
    with st.container(border = True):
        df = pd.read_csv(uploaded_file)
        with st.spinner('Running...'):
            #explainer = shap.Explainer(model, df[features])
            #shap_values = explainer(df[features])
            #st_shap(shap.plots.beeswarm(shap_values))

            tree_explainer = shap.TreeExplainer(model)
            tree_shap_values = tree_explainer.shap_values(df[features])

            st_shap(shap.summary_plot(tree_shap_values, df[features]))

            for name in features:
                st_shap(shap.dependence_plot(name, tree_shap_values, df[features], display_features=df[features]))



st.write("#")
st.markdown(TITLE_TEMPLATE.format("4- Features Importance"), unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax = lgb.plot_importance(model, importance_type="split", figsize=(4,3), title="LightGBM Feature Importance (Split)", ax = ax)
    ax.tick_params(labelsize=8)
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    #fig = ax.get_figure()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax = lgb.plot_importance(model, importance_type="gain", figsize=(4,3), title="LightGBM Feature Importance (Gain)", ax = ax)
    #fig = ax.get_figure()
    ax.tick_params(labelsize=8)
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    st.pyplot(fig)

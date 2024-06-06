import os
import sys
from datetime import datetime
import pandas as pd
import streamlit as st
from pickle import dump

sys.path.append("./code")
from train import train_model, save_trained_model
from tools import extract_scores, get_hist_model_scores, get_features_from_last_model

st.set_page_config(layout="wide")

TITLE_TEMPLATE = "<h4 style='text-align: center;font-weight: bold'>{}</h4>"



metadata = get_features_from_last_model("./data/models")

st.markdown(TITLE_TEMPLATE.format("1- Upload Training Dataset"), unsafe_allow_html=True)
with st.container(border = True):
    uploaded_file = st.file_uploader("Upload Dataset", type={"csv", "txt"},  label_visibility = "collapsed")

st.write("#")
if uploaded_file:
    st.markdown(TITLE_TEMPLATE.format("2- Select Features and Target"), unsafe_allow_html=True)
    with st.container(border = True):
        default_target = None
        default_features = []
        if (metadata is not None):
            default_target = metadata.get("TARGET")
            if (metadata.get("FEATURES") is not None):
                default_features = metadata.get("FEATURES")
            
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.session_state["features"]  = st.multiselect("Select features",
                                                       st.session_state["df"].columns,
                                                       default_features)
        other_cols = [c for c in st.session_state["df"].columns if c not in st.session_state["features"]]
        target_index = 0 if (default_target not in other_cols) else other_cols.index(default_target)
        st.session_state["target"] = st.selectbox("Select target", index = target_index, options = other_cols)
    
    if ((len(st.session_state["features"]) > 0) and (st.session_state["target"] is not None)):
        st.write("#")
        st.markdown(TITLE_TEMPLATE.format("3- Train Model"), unsafe_allow_html=True)
        with st.container(border = True):
            train_btn = st.button("Train Model")
            if train_btn:
                with st.spinner('Training...'):
                    st.session_state["train_models"] = train_model(st.session_state["df"], 
                                                                     st.session_state["features"],
                                                                     st.session_state["target"])
                    st.session_state["model_id"] = "model_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.success('Model Trained!')

        st.write("#")
        st.markdown(TITLE_TEMPLATE.format("4- Evaluate Model"), unsafe_allow_html=True)
        with st.container(border = True):
            evaluate_btn = st.button("Evaluate Model")
            if evaluate_btn:
                df_actual_model_scores = pd.DataFrame(extract_scores(st.session_state["train_models"]["MODEL_GS"]),
                                                      index = [st.session_state["model_id"]])
                df_actual_model_scores = round(df_actual_model_scores*100,2)
                score_metrics = df_actual_model_scores.columns
                df_actual_model_scores["MODEL_TYPE"] = "Actual"
                df_hist_scores = get_hist_model_scores("./data/models")
                df_hist_scores = round(df_hist_scores*100,2)
                df_hist_scores["MODEL_TYPE"] = "Historical"
                df_all_scores = pd.concat([df_actual_model_scores, df_hist_scores])
                #st.write(df_all_scores)
                df_all_scores.index = [datetime.strptime(i.replace("model_", ""), "%Y%m%d_%H%M%S") for i in df_all_scores.index]
                #df_all_scores = df_all_scores.reset_index(names = ["MODEL"])

                cols = st.columns([1]*len(score_metrics))
                for i, m in enumerate(score_metrics):
                    with cols[i]:
                        st.markdown("<h6 style='text-align: center;'>{}</h6>".format(m), unsafe_allow_html=True)
                        st.area_chart(data = df_all_scores[m], color = ["#00B2A9"], height = 200)
        st.write("#")
        st.markdown(TITLE_TEMPLATE.format("5- Save Model"), unsafe_allow_html=True)
        with st.container(border = True):
            save_model_btn = st.button("Save Model")
            if save_model_btn:
                with st.spinner('Saving...'):
                    model_folder = os.path.join("./data/models", st.session_state["model_id"])
                    if not os.path.exists(model_folder):
                        os.makedirs(model_folder)
                    save_trained_model(st.session_state["train_models"], model_folder)
                    st.success('Model Saved!')

            



